"""Static environment definitions and policy-safe runtime descriptors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


ENVIRONMENT_DESCRIPTOR_SCHEMA = "ora.environment-runtime-descriptor.v1"

# Environments whose sparse motion codecs passed a source-bound scripted
# motion qualification (RoboTwin: the v8 gate family behind
# ora.robotwin-motion-transaction-qualification.v1). This table
# is the single admission source for both the environment descriptor and
# RunSpec validation. An environment joins a codec row only in the same
# change as its own pinned-source scripted qualification record — never from
# task pressure alone. Each record below is a run output kept outside the
# source tree and named here by its sha256.
# libero_pro joined motion_action_v3 with a passing 14-case sparse-motion
# qualification (record sha256
# 6e6e75b2cf0d630cac5aad4c71f51ab9df3516fcedc66a16b329d43781596dba).
# The current LIBERO route set (settle, staged contact approach, direct
# driver) passed the same 14 cases (record sha256
# 86fda296ea679b84b0b558433ca17a7179bdf14d0907656484e3765981bf5b5f).
# The shared planner, depth collision world and timed routes passed the same
# 14 cases through ``ora qualify libero motion`` (record sha256
# 944c9493ccde5d6262a8317a3b77b72f15adc2a20b12a74166535022f8dd7b01).
# The jaw-twin retry of a wrist refusal and the planner's straight-line and
# turn-ranked answers to a wrist detour passed the same 14 cases (record sha256
# 7021638995a773697e349d4f88f498bcdcfecd1f87eb5933dc6ab5b615fa4d53).
# Motion profile v3 (the global route's JOINT_POSITION controller at kp 50 with
# a 0.2 rad reach and a per-knot velocity lead, pacing at 0.55 rad/s, a terminal
# settle that ends at rest inside the local bound) passed the same 14 cases with
# no torque clipped (record sha256
# 621a2f62229ae10366b570aa149d5b2faaf01c72404d67294e67c094ecf079f1).
# Motion profile v4 (v3 plus the straight OSC line for a clear, nearly straight
# free-space plan) passed the same 14 cases, the free-space cases on the line
# (record sha256
# 8eb177d79a528a64a01cde7ce43da95a08aad232ebf17ee3109c716fb630cb04).
# robodojo joined motion_action_v3 with a passing 24-case sparse-motion
# qualification; the current record is the 24-case pass (hang_mugs layout 0,
# two rounds) measured under the resolved robodojo_arx_x5_tabletop_v1
# envelope rather than the robodojo_compat +/-1.5 m bounds the first one
# used (record sha256
# a81d5892634c15c23d023e2faf516f63aa162a02fa7861cb61e8b9a4c96fa35a).
# RoboCasa365 joined on its scripted record (joint-position execution of
# admitted IK paths, gripper, shared wait, move_base translate/rotate/return,
# home recovery to joints, IK probe, verdict and close through the real worker,
# a 60-degree joint-space reorientation, a refused straight line into the
# cabinet front and a reach 0.30 m above the hand that admission meets by
# lifting the torso first; 19/19 checks) on OpenDrawer seed 0,
# target split (record sha256
# 2608ee8a734b7bfbe3d81592d4672265b9c1c2ec464a9804617405090627656b).
# The horizon fix (no control step once the episode must stop) re-passed the
# same 19 checks on OpenDrawer seed 0 (record sha256
# 50eb2aa2132a97c0e986e085449833d62a32dedaf25aaf56818a54db04e54148).
# End-effector paced arm paths re-passed the same 19 checks (record sha256
# b70015b7d32e66ae2c432c1b684a313d19b11c1afa639fe084d613bb32ba7fa5).
# Arm moves on the shared planner, IK and public depth world re-passed them
# (record sha256
# f2b0a5d4cdd30d56b9a24fd99a39f8a42180a2f2d481b1ba6f26eaa20d692c0a).
# The jaw-twin retry of a wrist refusal and the planner's answers to a wrist
# detour re-passed them (record sha256
# 8e9591654cca59a3a9e9d95f07a51021c921ef4ef7019b19c360d64eab4439b8).
# Base placement (stands judged by the move's own admission, the lift admitted
# on its whole timed path, move_base's footprint and the arm and held load it
# carries checked along the servo's path, the depth linked to a held load
# carried with it, split from a wall it touches), the settle and stall rules,
# and the planner's per-arm, prebuilt scenes re-passed them (record sha256
# 416f025c3c684c2d49fc601767d7227feb6e408fb7185a933e2c0d8d90465b75).
# RoboLab joined on its scripted record (absolute-IK Cartesian servo on the
# Robotiq TCP, gripper, shared wait, home recovery, verdict and close through
# the real Isaac Sim 5.1 worker, loaded geometry verified against the
# droid-franka-robotiq descriptor; 12/12 checks) on
# BananaInBowlTask seed 0 (record sha256
# 2b7d9487d24801993385d1d7ec68cf95620445bd50c9f322b3788a969e2c958d).
# The servo's per-tick end-effector speed cap re-passed the same 12 checks (record sha256
# 00d5641d4bb257f955e90edc347e6cbe4ca47252639a9e0bbdbd788838593d48).
# Shared planning, timed joint paths and home recovery on the planner passed 14/14 checks,
# including the planner's refusal of a pose out of reach (record sha256
# 3fc78338419e4bcbe4853fa54b097e25969db64155c26374086488f6ebaf620f).
# The jaw-twin retry of a wrist refusal and the planner's answers to a wrist
# detour re-passed the same 14 checks (record sha256
# 2962553016ee4018b78d0d67673e5acfeed5f4f4264290e543b0b9022bd797d0).
# The YAM station joined on its scripted record (warm-started IK joint schedules on the
# i2rt station model, every tick checked for joint limits, table clearance and inter-arm
# distance; precision, free-space and simultaneous two-arm moves, both grippers both ways,
# shared wait, home recovery, the refusals of an unreachable target and of two wrists
# brought 1.2 cm apart, null verdict and close through the real worker; 15/15 checks) on
# the simulated sim_pick_place cell, i2rt fe78bae (record sha256
# 44713149ec92bd8f8af06667c1d7beddcf0f0dd6061b0e6156647ac159f402b9).
# The real rig executes the same plans; its read and on modes add no planner.
SPARSE_MOTION_QUALIFIED_ENVIRONMENTS: dict[str, tuple[str, ...]] = {
    "motion_action_v3": ("robotwin", "libero_pro", "robodojo", "robocasa", "robolab", "yam"),
}
# The physics devices each (codec, environment) row's scripted record was
# measured on, for environments whose runtime declares ``physics_device``.
# Both RoboDojo records above ran on ``cuda:0``. A CPU-physics RoboDojo
# runtime (benchmarks/robodojo_physics.py, 2026-10-04) changes the broadphase,
# tensor API and USD/Fabric path the motion contract was measured through, so
# it joins only with its own passing scripted record, cited beside this row in
# the same change. RunSpec resolution refuses it until then
# (rule=sparse_motion_physics_unqualified); qualification probes that bypass
# the RunSpec (card, scene smoke, the scripted gate itself) are unaffected.
# The CPU row's record is the 24-case pass (swap_blocks layout 0, two rounds,
# physics_device cpu; record sha256
# da33671848505ae787a3a5db5b4cf4c5557d960cd15fe21084433698ddd15ec2).
SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES: dict[tuple[str, str], tuple[str, ...]] = {
    ("motion_action_v3", "robodojo"): ("cuda:0", "cpu"),
}
SPARSE_MOTION_PHYSICS_RULE = "sparse_motion_physics_unqualified"


def sparse_motion_physics_refusal(codec: str, runtime: Mapping[str, Any]) -> str | None:
    """Why ``runtime``'s physics device has no scripted motion record for ``codec``, or None."""

    qualified = SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES.get((codec, str(runtime.get("kind"))))
    device = runtime.get("physics_device")
    if qualified is None or device is None or device in qualified:
        return None
    return (
        f"sparse motion codec {codec} has no source-bound motion qualification for "
        f"{runtime.get('kind')} on physics device {device}; qualified: {', '.join(qualified)}; "
        f"rule={SPARSE_MOTION_PHYSICS_RULE}"
    )


ENVIRONMENT_DESCRIPTOR_V2_SCHEMA = "ora.environment-runtime-descriptor.v2"
#: Backends whose sparse v3 executor implements the shared ``wait`` hold
#: (LIBERO: ``backends/libero_pro/wait.py``, one tick per control step).
WAIT_ENVIRONMENTS = ("libero_pro", "robocasa", "robodojo", "robolab", "robotwin", "yam")
#: Backends with a holonomic mobile base: motion v3 adds ``move_base``.
MOBILE_BASE_ENVIRONMENTS = ("robocasa",)


class EnvironmentDefinitionError(RuntimeError):
    """A runtime cannot be resolved to one qualified environment definition."""


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True)
class EnvironmentRuntimeDescriptor:
    environment_id: str
    environment_version: str
    transport: str
    embodiment_id: str
    robot_models: tuple[str, ...]
    controller: str
    public_cameras: tuple[str, ...]
    frames: tuple[str, ...]
    units: Mapping[str, str]
    action_modes: tuple[str, ...]
    observation_schema: str
    action_schemas: tuple[str, ...]
    revisions: Mapping[str, str]
    capabilities: Mapping[str, bool]
    controlled_end_effectors: Mapping[str, Mapping[str, str]] | None = None
    control_groups: tuple[Mapping[str, Any], ...] = ()
    camera_roles: Mapping[str, str] | None = None
    schema: str = ENVIRONMENT_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in {
            ENVIRONMENT_DESCRIPTOR_SCHEMA, ENVIRONMENT_DESCRIPTOR_V2_SCHEMA,
        }:
            raise EnvironmentDefinitionError("environment descriptor schema is invalid")
        strings = (
            self.environment_id,
            self.environment_version,
            self.transport,
            self.embodiment_id,
            self.controller,
            self.observation_schema,
        )
        if not all(isinstance(value, str) and value for value in strings):
            raise EnvironmentDefinitionError("environment descriptor identity is invalid")
        if not self.public_cameras or not all(self.public_cameras):
            raise EnvironmentDefinitionError("environment descriptor requires public cameras")
        object.__setattr__(self, "units", _mapping(self.units))
        object.__setattr__(self, "revisions", _mapping(self.revisions))
        object.__setattr__(self, "capabilities", _mapping(self.capabilities))
        if self.schema == ENVIRONMENT_DESCRIPTOR_V2_SCHEMA:
            effectors = self.controlled_end_effectors
            if not isinstance(effectors, Mapping) or not effectors:
                raise EnvironmentDefinitionError(
                    "v2 environment descriptor requires controlled end effectors"
                )
            if not isinstance(self.camera_roles, Mapping) or not self.camera_roles:
                raise EnvironmentDefinitionError(
                    "v2 environment descriptor requires camera roles"
                )
            object.__setattr__(self, "controlled_end_effectors", _mapping(effectors))
            object.__setattr__(self, "camera_roles", _mapping(self.camera_roles))
            object.__setattr__(self, "control_groups", tuple(
                _mapping(group) for group in self.control_groups
            ))

    def as_dict(self) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "transport": self.transport,
            "embodiment_id": self.embodiment_id,
            "robot_models": list(self.robot_models),
            "controller": self.controller,
            "public_cameras": list(self.public_cameras),
            "frames": list(self.frames),
            "units": dict(self.units),
            "action_modes": list(self.action_modes),
            "observation_schema": self.observation_schema,
            "action_schemas": list(self.action_schemas),
            "revisions": dict(self.revisions),
            "capabilities": dict(self.capabilities),
        }
        if self.schema == ENVIRONMENT_DESCRIPTOR_V2_SCHEMA:
            result.update({
                "controlled_end_effectors": {
                    name: dict(record)
                    for name, record in (self.controlled_end_effectors or {}).items()
                },
                "control_groups": [dict(group) for group in self.control_groups],
                "camera_roles": dict(self.camera_roles or {}),
            })
        return result


@dataclass(frozen=True)
class EnvironmentDefinition:
    environment_id: str
    version: str
    transports: tuple[str, ...]
    embodiments: tuple[str, ...]
    observation_schema: str
    action_schemas: tuple[str, ...]
    capabilities: Mapping[str, bool]
    preflight_id: str
    worker_loader: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", _mapping(self.capabilities))

    def resolve_descriptor(
        self,
        runtime: Mapping[str, Any],
        *,
        action_codec: str | None = None,
        observed_public_context: Mapping[str, Any] | None = None,
    ) -> EnvironmentRuntimeDescriptor:
        transport = str(runtime.get("transport", "in_process_direct"))
        if transport not in self.transports:
            raise EnvironmentDefinitionError(
                f"unsupported transport for {self.environment_id}: {transport}"
            )
        public = observed_public_context or runtime.get("public_environment_context")
        if public is not None and not isinstance(public, Mapping):
            raise EnvironmentDefinitionError("public environment context must be an object")
        defaults = _DEFAULT_PUBLIC_CONTEXT[self.environment_id](runtime, action_codec)
        selected = defaults if public is None else dict(public)
        if selected.get("environment_id") != self.environment_id:
            raise EnvironmentDefinitionError("environment descriptor identity drift")
        embodiment = selected.get("embodiment_id")
        if embodiment not in self.embodiments:
            raise EnvironmentDefinitionError("environment descriptor embodiment drift")
        cameras = selected.get("public_cameras")
        models = selected.get("robot_models")
        if not isinstance(cameras, list) or not all(
            isinstance(item, str) and item for item in cameras
        ):
            raise EnvironmentDefinitionError("environment descriptor cameras are invalid")
        if not isinstance(models, list) or not all(
            isinstance(item, str) and item for item in models
        ):
            raise EnvironmentDefinitionError("environment descriptor robot models are invalid")
        controller = selected.get("controller")
        if not isinstance(controller, str) or not controller:
            raise EnvironmentDefinitionError("environment descriptor controller is invalid")
        codec = action_codec or str(runtime.get("action_codec", "runtime_action_v3"))
        if codec not in self.action_schemas:
            raise EnvironmentDefinitionError(
                f"unsupported action schema for {self.environment_id}: {codec}"
            )
        action_modes = (
            ("target_state",) if codec == "target_state_v4"
            else (
                "move_pose", "move_free_space", "set_gripper", "recover_motion",
            ) if codec == "motion_action_v3"
            else ("move_pose", "set_gripper")
        )
        if codec == "motion_action_v3" and self.environment_id in WAIT_ENVIRONMENTS:
            action_modes = (*action_modes, "wait")
        if codec == "motion_action_v3" and self.environment_id in MOBILE_BASE_ENVIRONMENTS:
            action_modes = (*action_modes, "move_base")
        descriptor_v2 = codec in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS
        if codec in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS and self.environment_id not in (
            SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[codec]
        ):
            raise EnvironmentDefinitionError(
                f"sparse motion codec {codec} has no source-bound motion "
                f"qualification for {self.environment_id}; qualified: "
                + ", ".join(SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[codec])
            )
        revisions = {
            key.removeprefix("required_"): value
            for key, value in runtime.items()
            if key.startswith("required_") and key.endswith("revision")
            and isinstance(value, str) and value
        }
        return EnvironmentRuntimeDescriptor(
            environment_id=self.environment_id,
            environment_version=self.version,
            transport=transport,
            embodiment_id=str(embodiment),
            robot_models=tuple(models),
            controller=controller,
            public_cameras=tuple(cameras),
            frames=("world", "camera", "end_effector"),
            units={"length": "metre", "angle": "radian"},
            action_modes=action_modes,
            observation_schema=self.observation_schema,
            action_schemas=(codec,),
            revisions=revisions,
            capabilities={
                **dict(self.capabilities),
                "isolated_worker": transport == "local_rpc",
                "policy_observer_camera": "observer_camera" in cameras,
            },
            controlled_end_effectors=(
                (
                    {
                        "left": {
                            "controlled_point": "end_effector",
                            "planner_route": "left",
                        },
                        "right": {
                            "controlled_point": "end_effector",
                            "planner_route": "right",
                        },
                    }
                    if self.environment_id == "robotwin"
                    # RoboDojo drives both ARX X5 arms at link6 with one SDK
                    # IK solve per arm inside a common native tick.
                    else {
                        "left": {
                            "controlled_point": "link6",
                            "planner_route": "left",
                        },
                        "right": {
                            "controlled_point": "link6",
                            "planner_route": "right",
                        },
                    }
                    if self.environment_id == "robodojo"
                    # The YAM station plans both arms at their tcp sites and
                    # merges the two joint schedules tick by tick.
                    else {
                        "left": {
                            "controlled_point": "tcp",
                            "planner_route": "left",
                        },
                        "right": {
                            "controlled_point": "tcp",
                            "planner_route": "right",
                        },
                    }
                    if self.environment_id == "yam"
                    # LIBERO drives one right Panda arm at the RoboSuite
                    # grip site; the v2 descriptor states that honestly
                    # instead of inheriting the dual-arm shape.
                    else {
                        "right": {
                            "controlled_point": "grip_site",
                            "planner_route": "right",
                        },
                    }
                )
                if descriptor_v2 else None
            ),
            control_groups=(
                ({
                    "group_id": "aloha_dual_arm",
                    "members": ["left", "right"],
                    "synchronous_pose_execution": True,
                },)
                if descriptor_v2 and self.environment_id == "robotwin"
                else ({
                    "group_id": "arx_x5_dual_arm",
                    "members": ["left", "right"],
                    "synchronous_pose_execution": True,
                },)
                if descriptor_v2 and self.environment_id == "robodojo"
                else ({
                    "group_id": "yam_dual_arm",
                    "members": ["left", "right"],
                    "synchronous_pose_execution": True,
                },)
                if descriptor_v2 and self.environment_id == "yam"
                else ()
            ),
            camera_roles=(
                {name: role for name, role in {
                    "observer_camera": "fixed_third_person_workspace",
                    "head_camera": "fixed_overhead",
                    "front_camera": "fixed_front",
                    "left_camera": "left_wrist",
                    "right_camera": "right_wrist",
                    "agentview": "fixed_third_person_workspace",
                    "wrist": "right_wrist",
                    "head": "fixed_third_person_workspace",
                    "over_shoulder_left": "fixed_third_person_workspace",
                    "left_wrist": "left_wrist",
                    "right_wrist": "right_wrist",
                    # The YAM station's fixed overhead and front cameras and its two wrists.
                    "overhead": "fixed_overhead",
                    "front": "fixed_front",
                    "wrist_left": "left_wrist",
                    "wrist_right": "right_wrist",
                }.items() if name in cameras}
                if descriptor_v2 else None
            ),
            schema=(
                ENVIRONMENT_DESCRIPTOR_V2_SCHEMA
                if descriptor_v2 else ENVIRONMENT_DESCRIPTOR_SCHEMA
            ),
        )


def _cameras(runtime: Mapping[str, Any], default: str) -> list[str]:
    raw = runtime.get("policy_cameras")
    if isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
        return list(raw)
    return [str(runtime.get("main_camera", default))]


def _synthetic(
    runtime: Mapping[str, Any], action_codec: str | None,
) -> dict[str, Any]:
    return {
        "environment_id": "synthetic",
        "embodiment_id": "synthetic_target_state",
        "robot_models": ["SyntheticTargetState"],
        "controller": action_codec or "target_state_v4",
        "public_cameras": _cameras(runtime, "head_camera"),
    }


def _robotwin(
    runtime: Mapping[str, Any], action_codec: str | None,
) -> dict[str, Any]:
    return {
        "environment_id": "robotwin",
        "embodiment_id": "aloha_agilex",
        "robot_models": ["ALOHA-AgileX"],
        "controller": action_codec or "runtime_action_v3",
        "public_cameras": _cameras(runtime, "head_camera"),
    }


def _libero(
    runtime: Mapping[str, Any], action_codec: str | None,
) -> dict[str, Any]:
    cameras = (
        _cameras(runtime, "agentview")
        if isinstance(runtime.get("policy_cameras"), list)
        else ["agentview", "wrist"]
    )
    return {
        "environment_id": "libero_pro",
        "embodiment_id": "franka_panda",
        "robot_models": ["Panda"],
        "controller": (
            "CUROBO_JOINT_POSITION+OSC_POSE"
            if action_codec == "motion_action_v3"
            else "OSC_POSE"
        ),
        "public_cameras": cameras,
    }


def _robodojo(runtime: Mapping[str, Any], action_codec: str | None) -> dict[str, Any]:
    from .benchmarks.robodojo import public_context

    return public_context()


def _robolab(runtime: Mapping[str, Any], action_codec: str | None) -> dict[str, Any]:
    from .benchmarks.robolab import public_context

    return public_context()


def _yam(runtime: Mapping[str, Any], action_codec: str | None) -> dict[str, Any]:
    from .benchmarks.yam import public_context

    return public_context()


def _robocasa(runtime: Mapping[str, Any], action_codec: str | None) -> dict[str, Any]:
    from .benchmarks.robocasa import public_context

    return public_context()


_DEFAULT_PUBLIC_CONTEXT = {
    "synthetic": _synthetic,
    "robotwin": _robotwin,
    "libero_pro": _libero,
    "robodojo": _robodojo,
    "robocasa": _robocasa,
    "robolab": _robolab,
    "yam": _yam,
}


_DEFINITIONS = {
    "yam": EnvironmentDefinition(
        environment_id="yam", version="v1", transports=("local_rpc",),
        embodiments=("yam_station_linear_4310_d405",), observation_schema="ora-observation.v1",
        # motion_action_v3: warm-started IK along sampled Cartesian paths, every tick
        # checked for joint limits, table clearance and inter-arm distance before the
        # arms move; on the simulated station model or the real arms (rig_mode).
        # A real cell has no evaluator, so the official verdict is null.
        action_schemas=("motion_action_v3",),
        capabilities={"rgb": True, "depth": True, "official_verdict": False, "isolated_worker": True},
        preflight_id="yam_v1",
        worker_loader="ora.environment.backends.yam.session:YamSession",
    ),
    "robolab": EnvironmentDefinition(
        environment_id="robolab", version="v1", transports=("local_rpc",),
        embodiments=("droid_franka_robotiq",), observation_schema="ora-observation.v1",
        # motion_action_v3: absolute differential-IK Cartesian servo on the
        # Robotiq TCP; gated by SPARSE_MOTION_QUALIFIED_ENVIRONMENTS with its
        # scripted motion record cited there.
        action_schemas=("motion_action_v3",),
        capabilities={"rgb": True, "depth": True, "official_verdict": True, "isolated_worker": True},
        preflight_id="robolab_v1",
        worker_loader="ora.environment.backends.robolab.session:RobolabSession",
    ),
    "robocasa": EnvironmentDefinition(
        environment_id="robocasa", version="v1", transports=("local_rpc",),
        embodiments=("panda_omron",), observation_schema="ora-observation.v1",
        # motion_action_v3: joint-position execution of admitted IK paths for the arm plus the base
        # servo behind move_base; gated by SPARSE_MOTION_QUALIFIED_ENVIRONMENTS
        # with its scripted motion record cited there.
        action_schemas=("motion_action_v3",),
        capabilities={"rgb": True, "depth": True, "official_verdict": True,
                      "isolated_worker": True, "mobile_base": True},
        preflight_id="robocasa_v1",
        worker_loader="ora.environment.backends.robocasa.session:RoboCasaSession",
    ),
    "robodojo": EnvironmentDefinition(
        environment_id="robodojo", version="v1", transports=("local_rpc",),
        embodiments=("arx_x5",), observation_schema="ora-observation.v1",
        # motion_action_v3 is implemented by the backend (SDK IK plus
        # joint interpolation, an explicit non-parity global route) and stays
        # gated by SPARSE_MOTION_QUALIFIED_ENVIRONMENTS until its scripted
        # motion qualification record is cited beside that table.
        action_schemas=("runtime_action_v3", "motion_action_v3"),
        capabilities={"rgb": True, "depth": True, "official_verdict": True, "isolated_worker": True},
        preflight_id="robodojo_v1",
        worker_loader="ora.environment.backends.robodojo.session:RobodojoSession",
    ),
    "synthetic": EnvironmentDefinition(
        environment_id="synthetic",
        version="v1",
        transports=("in_process_direct",),
        embodiments=("synthetic_target_state",),
        observation_schema="ora-observation.v1",
        action_schemas=("runtime_action_v3", "target_state_v4"),
        capabilities={
            "rgb": True, "depth": True, "official_verdict": False,
            "isolated_worker": False,
        },
        preflight_id="synthetic_v1",
        worker_loader="ora.environment.synthetic:SyntheticSession",
    ),
    "robotwin": EnvironmentDefinition(
        environment_id="robotwin",
        version="v1",
        transports=("local_rpc",),
        embodiments=("aloha_agilex",),
        observation_schema="ora-observation.v1",
        action_schemas=(
            "runtime_action_v3", "target_state_v4", "motion_action_v3",
        ),
        capabilities={
            "rgb": True, "depth": True, "official_verdict": True,
            "isolated_worker": True,
        },
        preflight_id="robotwin_v1",
        worker_loader=(
            "ora.environment.backends.robotwin.session:"
            "RoboTwinLiveSession"
        ),
    ),
    "libero_pro": EnvironmentDefinition(
        environment_id="libero_pro",
        version="v1",
        transports=("local_rpc",),
        embodiments=("franka_panda",),
        observation_schema="ora-observation.v1",
        # motion_action_v3 is implemented by the backend
        # but remains gated by SPARSE_MOTION_QUALIFIED_ENVIRONMENTS until the
        # LIBERO scripted motion qualification passes on pinned source.
        action_schemas=(
            "runtime_action_v3", "target_state_v4", "motion_action_v3",
        ),
        capabilities={
            "rgb": True, "depth": True, "official_verdict": True,
            "isolated_worker": True,
        },
        preflight_id="libero_pro_v1",
        worker_loader=(
            "ora.environment.backends.libero_pro.session:LiberoProSession"
        ),
    ),
}


def environment_definitions() -> tuple[EnvironmentDefinition, ...]:
    return tuple(_DEFINITIONS[key] for key in sorted(_DEFINITIONS))


def environment_definition(environment_id: str) -> EnvironmentDefinition:
    try:
        return _DEFINITIONS[environment_id]
    except KeyError as exc:
        raise EnvironmentDefinitionError(
            f"unregistered environment: {environment_id!r}"
        ) from exc


def resolve_runtime_descriptor(
    runtime: Mapping[str, Any],
    *,
    action_codec: str | None = None,
    observed_public_context: Mapping[str, Any] | None = None,
) -> EnvironmentRuntimeDescriptor:
    kind = runtime.get("kind")
    if not isinstance(kind, str):
        raise EnvironmentDefinitionError("runtime.kind must identify an environment")
    return environment_definition(kind).resolve_descriptor(
        runtime,
        action_codec=action_codec,
        observed_public_context=observed_public_context,
    )


def validate_runtime_extension(runtime: Mapping[str, Any]) -> None:
    """Validate environment-owned optional fields through host-safe contracts."""

    if runtime.get("kind") == "robodojo":
        from .benchmarks.robodojo import RoboDojoBenchmarkError, validate_runtime_extension as validate_robodojo
        if runtime.get("play_scene_set") is not None:
            # One of ORA's own play scenes: the official-task rule admits no such task.
            from .benchmarks.robodojo_play_scenes import validate_play_runtime as validate_robodojo
        try:
            validate_robodojo(runtime)
        except RoboDojoBenchmarkError as exc:
            raise EnvironmentDefinitionError(str(exc)) from exc
        return
    if runtime.get("kind") == "robolab":
        from .benchmarks.robolab import RoboLabBenchmarkError, validate_runtime_extension as validate_robolab
        try:
            validate_robolab(runtime)
        except RoboLabBenchmarkError as exc:
            raise EnvironmentDefinitionError(str(exc)) from exc
        return
    if runtime.get("kind") == "yam":
        from .benchmarks.yam import YamBenchmarkError, validate_runtime_extension as validate_yam
        try:
            validate_yam(runtime)
        except YamBenchmarkError as exc:
            raise EnvironmentDefinitionError(str(exc)) from exc
        return
    if runtime.get("kind") == "robocasa":
        from .benchmarks.robocasa import RoboCasaBenchmarkError, validate_runtime_extension as validate_robocasa
        try:
            validate_robocasa(runtime)
        except RoboCasaBenchmarkError as exc:
            raise EnvironmentDefinitionError(str(exc)) from exc
        return
    if runtime.get("kind") != "libero_pro":
        return
    from .benchmarks.libero_pro import (
        LiberoProBenchmarkError,
        validate_runtime_extension as validate_libero_pro_extension,
    )

    try:
        validate_libero_pro_extension(runtime)
    except LiberoProBenchmarkError as exc:
        raise EnvironmentDefinitionError(str(exc)) from exc
