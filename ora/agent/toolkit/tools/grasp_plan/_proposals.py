"""Tool-local hybrid policy: geometric-first or contact-first, learned fill, compile, lift and IK order."""
from typing import Any

from ...lib import compilation, grasp
from ...lib.values import JsonValue
from ._learned import learned
from ._topdown import MODEL, PcaTopDownGraspProposer, TargetInsufficient
from ._diagnostics import hybrid_grasp_diagnostics
from ._lift import split_liftable
from ._selection import apply_candidate_selection

#: A hybrid whose top-down family is a fallback, shown only when the learned one returned no candidate at all (the Panda).
#: A context without ``selection``, every RoboDojo and sealed one, fills up to two top-down candidates first.
CONTACT_FIRST = "topdown_only_without_contact_v1"


def proposals(api, *, scene, target, transform, target_ref, workspace, maximum, config, compiler, selection):
    pool_limit = maximum if selection is None else selection['identity']['pool_limit']
    learned_backend = config['contact_backend']
    pool = grasp.candidates(api, scene=scene, target=target, camera_to_world=transform,
        workspace_bounds_m=workspace, max_candidates=learned_backend['configuration']['raw_candidate_limit'],
        forward_passes=learned_backend['configuration']['forward_passes'])
    learned_rows, learned_diagnostics = learned(pool, camera_to_world=transform.matrix.numpy(),
        target_ref=target_ref, workspace_bounds_m=workspace.json(), maximum=pool_limit,
        backend=learned_backend, compiler=compiler)
    service_identity = learned_diagnostics.pop('_service_identity', None)
    input_regime = learned_diagnostics.pop('input_regime', None)
    if config.get('topdown_backend') is not None:
        settings = config['topdown_backend']
        topdown = PcaTopDownGraspProposer(backend_id=settings['id'], backend_version=settings['version'],
            **{k:v for k,v in settings['configuration'].items() if k != 'proposal_family'})
        contact_first = config.get('selection') == CONTACT_FIRST
        try:
            geometric, geometric_diagnostics = topdown.propose(scene_points_camera_m=scene.xyz_m.numpy(),
                target_points_camera_m=target.xyz_m.numpy(), camera_to_world=transform.matrix.numpy(),
                target_ref=target_ref, workspace_bounds_m=workspace.json(), max_candidates=pool_limit,
                compiler_identity=compiler)
        except TargetInsufficient as error:
            if not contact_first:
                raise
            # A target too sparse for the fallback family costs only that family.
            geometric, geometric_diagnostics = [], dict(model=dict(MODEL), rejected=[], warning=str(error))
        candidates = hybrid(learned_rows, geometric, pool_limit, contact_first=contact_first)
        diagnostics = hybrid_grasp_diagnostics(learned_diagnostics=learned_diagnostics,
            geometric_diagnostics=geometric_diagnostics, learned=learned_rows, geometric=geometric, selected=candidates)
    else:
        candidates, diagnostics = learned_rows, learned_diagnostics
    raw, compiled, rejected = [], [], []
    for candidate in candidates:
        try:
            result = compilation.compile_contact(JsonValue.of(candidate), JsonValue.of(compiler),
                public_contact_pose=config['public_contact_pose']).json()
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(dict(candidate_id=str(candidate.get('candidate_id', 'unknown')), reason=str(error)))
        else:
            raw.append(dict(candidate, rank=len(raw)))
            compiled.append(dict(result, rank=len(compiled)))
    if config.get('lift_m') is not None:
        raw, compiled, unliftable = split_liftable(raw, compiled, config['lift_m'], workspace.json())
        rejected.extend(unliftable)
    diagnostics['compiler_rejected'] = rejected
    raw, compiled, selected = apply_candidate_selection(raw, compiled, selection, maximum)
    diagnostics['compiler_returned_candidate_count'] = len(compiled)
    return dict(candidates=raw, compiled_candidates=compiled, diagnostics=diagnostics,
        response_service_identity=service_identity, input_regime=input_regime, selection_record=selected), pool


def hybrid(learned, geometric, max_candidates, *, contact_first=False):
    shown = [] if contact_first and learned else geometric
    selected_geometric_count = 0 if contact_first else min(2, max_candidates, len(shown))
    selected = [dict(item) for item in shown[:selected_geometric_count]]
    selected.extend(
        dict(item) for item in learned[: max_candidates - len(selected)]
    )
    if len(selected) < max_candidates:
        selected.extend(
            dict(item)
            for item in shown[
                selected_geometric_count:
                selected_geometric_count + max_candidates - len(selected)
            ]
        )
    seen: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for candidate in selected:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidate["rank"] = len(deduplicated)
        deduplicated.append(candidate)
    return deduplicated
