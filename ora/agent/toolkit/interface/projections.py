"""Field-by-field public projections. Unknown worker fields never pass through."""
from __future__ import annotations

from collections.abc import Mapping
import json

from ora.contracts.planning_diagnostics import IK_CONSTRAINT_KEYS, public_planning_diagnostics
from ora.contracts.reach_check import RESULT_FIELDS, ROW_FIELDS

from .types import InterfaceError, PublicData, canonical


def fields(value: Mapping, names: str, *, structured: tuple[str, ...] = ()) -> dict:
    """Select declared JSON-valued semantic fields; no object serialization."""
    def public_leaf(item):
        if isinstance(item, dict):
            raise InterfaceError("undeclared nested object in public scalar field")
        if isinstance(item, list):
            for child in item:
                public_leaf(child)
    result = {}
    for name in names.split():
        if name in value:
            if name not in structured:
                public_leaf(value[name])
            result[name] = json.loads(canonical(value[name]))
    return result


def motion_probe(value: Mapping) -> PublicData:
    data = fields(value, "schema observation_id world_mutation_sequence variant_policy robot_decisions_consumed world_state_changed")
    data["results"] = []
    for result in value["results"]:
        row = fields(result, "query_id arm reachable selected_variant_id plan_ref selected_target_pose_xyzw")
        row["variants"] = []
        for variant in result["variants"]:
            entry = fields(variant, "variant_id reachable target_pose_xyzw trajectory_steps trajectory_duration_s joint_delta_rad plan_ref")
            for key, allowed in (("failure", "code message"), ("endpoint_error", "translation_m orientation_rad")):
                if key in variant:
                    entry[key] = fields(variant[key], allowed)
            planner = variant.get("planner", {})
            projected = fields(planner, "called status stage valid_query attempts trajopt_attempts used_graph goalset_index path_steps interpolation_dt_s planner parity")
            # A variant the planner failed carries the public planner error, an object on RoboTwin and LIBERO-Pro
            # (``planner_public_summary``) and absent or null on RoboDojo.
            if isinstance(planner.get("error"), Mapping):
                projected["error"] = fields(planner["error"], "stage type message")
            elif "error" in planner:
                projected.update(fields(planner, "error"))
            for key, allowed in (("errors", "position rotation cspace"),
                                 ("timings_s", "solve ik graph trajopt finetune total"),
                                 ("constraint_rejection_counts", " ".join(IK_CONSTRAINT_KEYS)),
                                 ("ik_validation_artifact", "path schema sha256")):
                if key in planner:
                    projected[key] = fields(planner[key], allowed)
            if "planning_diagnostics" in planner:
                projected["planning_diagnostics"] = public_planning_diagnostics(planner["planning_diagnostics"])
            entry["planner"] = projected
            row["variants"].append(entry)
        data["results"].append(row)
    return PublicData.from_json(data)


def alignment(value: Mapping) -> PublicData:
    data = fields(value, "schema target_ref approach_side requested_arm motion_planner_called")
    data["candidates"] = []
    for row in value["candidates"]:
        entry = fields(row, "source_index arm reachable position_error_m orientation_error_rad iterations rejection_reason claim_boundary")
        entry["align_pose"] = fields(row["align_pose"], "position_m orientation_xyzw")
        data["candidates"].append(entry)
    return PublicData.from_json(data)


def reach(value: Mapping) -> PublicData:
    data = fields(value, " ".join(sorted(RESULT_FIELDS - {"results"})))
    data["results"] = [fields(row, " ".join(sorted(ROW_FIELDS))) for row in value["results"]]
    return PublicData.from_json(data)


def kinematics(value: Mapping) -> PublicData:
    """The read-only kinematics answer (``ora.kinematics-result.v1``), field by field."""
    data = fields(value, "schema operation arm seed_joints_rad first_failure_index ik_model seeding total_s claim_boundary "
                  "ik_sampler_state warning "
                  "status first_blocked_segment segments_screened min_inter_arm_clearance_m "
                  "min_table_clearance_m limiting_link "
                  "reason message start_joints path_joints world_sha256 held_voxels fractions duration_s "
                  "ticks needed_ticks fraction_of_move slow_from_tick first_blocked_row penetration_m")
    for key in ("goal_refusal", "constraints", "speed", "wrist"):
        if isinstance(value.get(key), Mapping):
            data[key] = json.loads(canonical(value[key]))
    for key, allowed in (("timings_s", "ik fk screen"), ("limiting_links", "left right")):
        if isinstance(value.get(key), Mapping):
            data[key] = fields(value[key], allowed)
    if "rows" in value:
        data["rows"] = [fields(row, "index status joints_rad joint_step_rad position_error_m orientation_error_rad "
                               "limit_margin_rad canonical pitch_joint_wrap path valid_candidates pose_xyz_xyzw seeding",
                               structured=("seeding",))
                        for row in value["rows"]]
    return PublicData.from_json(data)


def receipt(value: Mapping) -> PublicData:
    """Public action outcome and measured robot state, not raw controller logs."""
    data = fields(value, "schema action action_id observation_id success accepted status reason error_code "
                  "world_mutation_sequence action_sequence robot_decisions_consumed physics_clock execution_status "
                  "executor motion_strategy failure_stage failure_code consumed native_action_budget native_actions "
                  "physics_steps sim_steps_total state_changed steps_used round_index started_unix_s finished_unix_s "
                  "target_arms terminal_after_action")
    for phase in ("before", "after"):
        if phase in value:
            data[phase] = {}
            for arm in ("left", "right"):
                if arm not in value[phase]:
                    continue
                measured = value[phase][arm]
                row = fields(measured, "arm_joint_qpos gripper gripper_command gripper_command_target "
                             "gripper_finger_qpos pose_frame pose_xyz_xyzw")
                if "gripper_feedback" in measured:
                    feedback = measured["gripper_feedback"]
                    row["gripper_feedback"] = fields(feedback, "schema drive_target_opening_normalized drive_target_source "
                                                     "effective_target_opening_normalized finger_lateral_offset_m "
                                                     "measured_opening_normalized measurement_source pose_hold_semantics "
                                                     "requested_command requested_opening_normalized")
                    if "arrival_check" in feedback:
                        row["gripper_feedback"]["arrival_check"] = fields(
                            feedback["arrival_check"], "comparison residual_normalized tolerance_normalized within_tolerance")
                data[phase][arm] = row
    if "arms" in value:
        data["arms"] = {}
        for arm in ("left", "right"):
            if arm not in value["arms"]:
                continue
            original = value["arms"][arm]
            row = fields(original, "targeted interaction plan_ref gripper_error")
            for key, allowed in (("inactive_arm_drift", "orientation_rad translation_m"),
                                 ("pose_error", "orientation_rad translation_m"),
                                 ("gripper_outcome", "state claim_boundary commanded_opening_normalized effective_target_opening_normalized "
                                  "finger_lateral_offset_m measured_finger_displacements_m measured_opening_normalized "
                                  "primary_finger_limit_holds_close sdk_drive_limit_normalized sdk_limited_drive_target_normalized "
                                  "seconds_used tick_period_s tick_unit ticks_budget ticks_used")):
                if key in original:
                    row[key] = None if original[key] is None else fields(original[key], allowed)
            data["arms"][arm] = row
    return PublicData.from_json(data)


def control_result(value: Mapping) -> PublicData:
    data = fields(value, "accepted directive action_id success outcome reason retryable task_success")
    if "_runtime" in value:
        data["_runtime"] = fields(value["_runtime"], "directive")
    if not data:
        raise InterfaceError("control adapter supplied no public result fields")
    return PublicData.from_json(data)
