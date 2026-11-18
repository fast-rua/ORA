"""Public before-call grants for a different tool; no final control/artifact scans."""

from types import SimpleNamespace

from ora.agent.composition.surface.tool_names import canonical as tool_name
from ora.agent.toolkit.interface.missing import missing_input, missing_request
from ora.agent.toolkit.interface.runtime import envelope
from ora.agent.toolkit.interface.types import (
    CorruptRecord,
    PublicData,
    canonical,
    decode,
    digest,
)
from ora.agent.toolkit.host.publication import published_manifest
from ora.agent.toolkit.host.runtime.tool_data import configuration
from .selection import read_local


def run_surface(source):
    """Frozen run definitions, including its explicit policy tool allowlist."""
    plan = (
        decode(read_local(source, "plan.json"))
        if (source / "plan.json").is_file()
        else {}
    )
    surface = plan.get("policy_composition", {}).get("tool_surface")
    if surface is None and (source / "compiled-policy/composition.json").is_file():
        surface = decode(read_local(source, "compiled-policy/composition.json")).get(
            "tool_surface"
        )
    if not isinstance(surface, dict) or not isinstance(
        surface.get("definitions"), list
    ):
        return None, None
    names = {tool_name(row["name"]) for row in surface["definitions"]}
    allowlist = plan.get("policy", {}).get("tool_allowlist")
    if allowlist is not None:
        names &= {tool_name(name) for name in allowlist}
    return surface, names


class Publications:
    def __init__(self, api, source, selected, events, surface):
        self.api, self.source, self.surface = api, source, surface
        self.current = api.context.json()["state"].get("observation")
        self.grants = (
            {} if not self.current else {self.current["observation_id"]: self.current}
        )
        self.steps = {}
        self.current_step = None
        for event in events:
            if (
                event["seq"] >= selected["proposal"]["seq"]
                or event["type"] != "fresh_observation_published"
            ):
                continue
            ref = event["payload"].get("state_ref")
            if not ref or not ref.get("path"):
                continue
            # Only an explicitly published, hash-bound state establishes a saved step.
            if not ref["path"].startswith("policy-state/steps/"):
                raise CorruptRecord("unexpected public-state reference")
            data = read_local(source, ref["path"])
            if digest(data) != ref["sha256"]:
                raise CorruptRecord("published state reference drift")
            state = decode(data)
            observed = state["observation"]
            identity = dict(
                observation_id=observed["id"],
                observation_sha256=observed["manifest"]["sha256"],
            )
            if identity["observation_id"] != event["payload"]["observation_id"]:
                raise CorruptRecord("publication/state observation mismatch")
            root, manifest, _ = published_manifest(api.store, identity)
            if manifest["round_index"] == api.context.json()["round_index"]:
                self.grants[identity["observation_id"]] = identity
                self.steps[str(state["step"])] = identity
                if identity == self.current:
                    self.current_step = state["step"]

    def read(self, method, request):
        identity = request["identity"]["parameters"]["observation"]
        if not identity or self.grants.get(identity.get("observation_id")) != identity:
            raise missing_request(
                request, reason="observation_not_published_at_boundary"
            )
        root, manifest, ref = published_manifest(self.api.store, identity)
        files = []
        paths = [("proprioception.json", None)]
        for camera in manifest.get("cameras", {}).values():
            paths.extend(
                (item["path"], item) for item in camera.get("files", {}).values()
            )
        for name, declared in paths:
            path = root / name
            if not path.is_file():
                continue
            media = (
                "image/png"
                if name.endswith(".png")
                else (
                    "application/x-npy" if name.endswith(".npy") else "application/json"
                )
            )
            item = self.api.store.reference(
                path.relative_to(self.source).as_posix(), media_type=media
            )
            if declared is not None and any(
                item[key] != declared[key] for key in ("sha256", "bytes")
            ):
                raise CorruptRecord("published camera artifact drift")
            files.append(item)
        value = dict(observation=identity, manifest=ref, files=files)
        if method == "proprioception":
            found = [
                item for item in files if item["path"].endswith("/proprioception.json")
            ]
            if len(found) != 1:
                raise missing_request(request, reason="proprioception_not_recorded")
            value = dict(proprioception=found[0])
        elif (
            self.surface is not None
            and "current_public_robot_facts_v1" in self.surface["capabilities"]
        ):
            from ora.agent.context.observation.contact_points import (
                project_policy_public_facts,
            )
            from ora.agent.composition.surface.view_config import (
                controlled_point_projection_delivery,
                gripper_command_delivery,
            )
            from ora.agent.error_classes import PublicFactsError
            from ora.agent.error_classes import ToolError

            view_config = self.surface["view_config"]
            facts = {}
            for view in manifest["cameras"]:
                try:
                    facts[view] = dict(
                        value=project_policy_public_facts(
                            observation=manifest,
                            observation_root=root,
                            view_config=view_config,
                            selected_views=[view],
                            snapshot_role="current",
                            gripper_command_delivered=gripper_command_delivery(
                                view_config
                            ),
                            controlled_point_projection=controlled_point_projection_delivery(
                                view_config
                            ),
                        )
                    )
                except (PublicFactsError, OSError) as error:
                    facts[view] = dict(
                        error=ToolError(
                            str(error), code="embodiment_binding", retryable=False
                        ).as_dict()
                    )
            value["public_robot_facts_by_view"] = facts
        response = envelope(
            method, request, public=PublicData.from_json(value, sources=[ref])
        )
        result = response.json()
        result["provenance"].update(
            reconstruction="published_boundary_artifact", exact_interface_bytes=False
        )
        return type(response)(canonical(result))

    def observe_configuration(self, arguments):
        if self.surface is None or self.current is None:
            raise missing_input(
                "tool_context",
                {"section": "observe"},
                self.api.context,
                reason="frozen_surface_or_observation_missing",
            )
        if arguments.get("history_point_ids"):
            raise missing_input(
                "tool_context",
                {"section": "observe.points"},
                self.api.context,
                reason="point_ledger_not_frozen_at_boundary",
            )
        _, manifest, _ = published_manifest(self.api.store, self.current)
        surface = self.surface
        capabilities, view = surface["capabilities"], surface["view_config"]
        result = dict(
            perception_feedback="public_rgbd_perception_feedback_v2" in capabilities,
            cameras=sorted(manifest["cameras"]),
            default_views=view["default_observe_views"],
            rgb_resolution=view["rgb_resolution"],
            points={},
            retention="public_rgbd_grasp_retention_recovery_v5" in capabilities,
            current_step=self.current_step,
            history=self.steps,
            public_facts="current_public_robot_facts_v1" in capabilities,
            measured_jaw="measured_jaw_separation_v1" in capabilities,
        )
        if self.steps:
            result["step_range"] = [
                min(map(int, self.steps)),
                max(map(int, self.steps)),
            ]
        elif arguments.get("step") is not None:
            raise missing_input(
                "tool_context",
                {"section": "observe.history", "step": arguments["step"]},
                self.api.context,
                reason="published_state_step_not_recorded",
            )
        return result


def complete_configuration(api, source, selected, events, surface, tool, arguments):
    """Only static surface/observation computations may fill an absent input section."""
    config = api.tool_data.context()
    publications = Publications(api, source, selected, events, surface)
    if (
        tool == "vision_ground"
        and "grounding_backend" not in config
        and surface is not None
    ):
        binding = api.bindings.get("ground", {})
        backend = SimpleNamespace(
            backend_id=binding.get("backend", "sam3_api"),
            backend_version="v1",
            configuration=lambda: dict(binding.get("configuration_identity") or {}),
        )
        frozen = SimpleNamespace(
            profile_id=surface["profile_id"],
            view_config=surface["view_config"],
            result_schemas=surface["result_schemas"],
            supports=lambda key: key in surface["capabilities"],
        )
        config.update(
            configuration(SimpleNamespace(surface=frozen, grounding_backend=backend))
        )
    if tool == "observe" and "observe" not in config:
        config["observe"] = publications.observe_configuration(arguments)
    sections = {
        "move_pose": "motion",
        "move_to": "motion",
        "grasp_plan": "grasp",
        "grasp_execute": "execution",
        "finish_episode": "lifecycle",
        "vision_ground": "grounding_backend",
    }
    required = sections.get(tool)
    if required is not None and required not in config:
        raise missing_input(
            "tool_context",
            {"tool": tool, "section": required},
            api.context,
            reason="configuration_not_frozen_at_boundary",
        )
    api.tool_data.configuration = canonical(config)
    return publications
