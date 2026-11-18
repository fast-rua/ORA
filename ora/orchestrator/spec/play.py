"""Resolve the optional free-play binding of a RunSpec (``ora play``).

A free-play RunSpec carries ``episode_mode: "play"`` (a benchmark one carries
no such key; its plan bytes are unchanged) and a ``play`` block. It loads a
qualified scene without its task: ``task_instruction`` is the session's
framing, the task contract is withheld and the policy is told nothing of the
scene's benchmark task. ``play.scene`` is the scene the run starts in
(``environment``, ``task``, ``seed`` and its ``qualification`` record, with
``scene_id``, ``label`` and the LIBERO suite and task id or the RoboTwin scene
configuration and, for a scene of ORA's own RoboTwin or RoboDojo scene sets,
its ``scene_set``, from ``play.scenes.build_play_spec``); ``play.scenes`` are the
scenes ``reset`` may load in turn, bound in the plan, starting with
``play.scene``. The plan runtime is the scene's play binding, which loads it
with the official predicate withheld (``play.binding.bind_play_runtime``).
``play.instruction_budgets`` bound each instruction and ``budgets`` the whole
session. ``play.context_images`` (default ``CONTEXT_IMAGES``) is how many of the
latest observations and tool results keep their images in a model request, and
with ``prune_at`` that the older ones leave in batches where an instruction begins.
``play.draw`` records how the scenes were drawn. Free play runs on the
Pi agent core: it is Pi's own agent (ora-pi/src/play/).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..episode import agent_core
from ..backends.registry import find
from ..play import binding
from .values import RunSpecError, _positive_int

PLAY_EXPERIMENT = "ora-play"
PLAY_SCHEMA = "ora.play-session.v1"
EPISODE_MODES = frozenset({"benchmark", "play"})
#: A free-play request carries the images of only this many latest observation messages and tool
#: results with images; older ones are one line of text each (ora-pi/src/play/context_images.ts).
#: Sent in full, a long session outgrew the provider's body limit (HTTP 413 at 50.8 MB).
#: ``prune_at``: the images all stay until an instruction begins with that many messages of a kind
#: still carrying theirs, then all but the latest ``observations`` / ``tool_results`` leave at once, so
#: the calls of one instruction share a prefix the provider can cache. A RunSpec that names the window
#: without it (and every plan resolved before it existed) keeps the window that slides at every call.
CONTEXT_IMAGES = {"observations": 3, "tool_results": 3, "prune_at": 20}
_IMAGE_WINDOW = ("observations", "tool_results")
MAX_SCENES = 32
_BUDGETS = ("max_robot_decisions", "max_model_calls", "wall_time_s")
_SCENE_FIELDS = ("environment", "task", "seed", "qualification")


def episode_mode(spec: Mapping[str, Any]) -> str:
    """``play`` for a free-play RunSpec or plan, else ``benchmark`` (the key's absence)."""

    mode = spec.get("episode_mode", "benchmark")
    if mode not in EPISODE_MODES:
        raise RunSpecError("episode_mode must be benchmark or play")
    if (mode == "play") != ("play" in spec):
        raise RunSpecError('a play block and episode_mode "play" go together')
    return mode


def _scene(value: Any, label: str, *, task: str) -> dict[str, Any]:
    # play.scenes' record, kept as it is: environment, task, seed and qualification are
    # required, and its id, label and backend identity (suite and task_id, task_config) come along.
    if not isinstance(value, Mapping) or not set(_SCENE_FIELDS) <= set(value):
        raise RunSpecError(f"{label} needs environment, task, seed and qualification")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RunSpecError(f"{label}.seed must be a non-negative integer")
    for field in ("environment", "task"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise RunSpecError(f"{label}.{field} must be non-empty text")
    if value["task"] != task:
        # restart_round changes the seed in the same worker, never the task.
        raise RunSpecError(f"{label}.task must be the plan's task ({task})")
    if not isinstance(value["qualification"], Mapping):
        raise RunSpecError(f"{label}.qualification must be an object (the record that admits the scene)")
    for field in ("scene_id", "label", "suite", "task_config", "scene_set"):
        text = value.get(field)
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise RunSpecError(f"{label}.{field} must be non-empty text")
    task_id = value.get("task_id")
    if task_id is not None and (isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0):
        raise RunSpecError(f"{label}.task_id must be a non-negative integer")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RunSpecError(f"{label} must be a JSON object") from exc


def resolve_play(
    spec: Mapping[str, Any], policy: Mapping[str, Any], runtime: Mapping[str, Any],
    budgets: Mapping[str, Any], *, seed: int, task: str,
) -> dict[str, Any] | None:
    """The normalized ``play`` block, or None for an ordinary run."""

    if episode_mode(spec) != "play":
        return None
    raw = spec["play"]
    if not isinstance(raw, Mapping) or set(raw) - {"schema", "scene", "scenes", "instruction_budgets", "draw",
                                                   "context_images", "cell_setup", "code_mode",
                                                   "request_priority", "pipeline_predict"}:
        raise RunSpecError("play takes schema, scene, scenes, instruction_budgets, context_images, draw, "
                           "cell_setup, code_mode, request_priority and pipeline_predict")
    if raw.get("schema", PLAY_SCHEMA) != PLAY_SCHEMA:
        raise RunSpecError(f"play.schema must be {PLAY_SCHEMA}")
    if spec.get("experiment") != PLAY_EXPERIMENT:
        raise RunSpecError(f"a free-play session runs under experiment {PLAY_EXPERIMENT}")
    if "operator_control" not in spec:
        raise RunSpecError("a free-play session needs operator_control (its instructions)")
    if "session" in spec:
        raise RunSpecError("a free-play session is one run; it takes no session block")
    if policy.get("kind") != "native_api":
        raise RunSpecError("a free-play session drives a native_api policy")
    if agent_core.selected(policy) != "pi":
        raise RunSpecError("a free-play session runs on the Pi agent core (it is Pi's own agent)")
    scene = _scene(raw.get("scene"), "play.scene", task=task)
    if scene["seed"] != seed:
        raise RunSpecError("play.scene.seed must be the plan's seed")
    scenes = raw.get("scenes", [scene])
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= MAX_SCENES:
        raise RunSpecError(f"play.scenes must list 1..{MAX_SCENES} scenes")
    normalized = [_scene(item, f"play.scenes[{index}]", task=task) for index, item in enumerate(scenes)]
    if normalized[0] != scene:
        raise RunSpecError("play.scenes[0] must be play.scene")
    _require_backend_scenes(runtime, normalized)
    instruction_budgets = raw.get("instruction_budgets")
    if not isinstance(instruction_budgets, Mapping) or set(instruction_budgets) != set(_BUDGETS):
        raise RunSpecError("play.instruction_budgets takes " + ", ".join(_BUDGETS))
    bounded = {}
    for field in _BUDGETS:
        bounded[field] = _positive_int(instruction_budgets[field], f"play.instruction_budgets.{field}")
        if bounded[field] > budgets[field]:
            raise RunSpecError(f"play.instruction_budgets.{field} exceeds the session's budgets.{field}")
    draw = raw.get("draw", {})
    if not isinstance(draw, Mapping) or not all(
        isinstance(key, str) and (value is None or isinstance(value, (str, int, float, bool)))
        for key, value in draw.items()
    ):
        raise RunSpecError("play.draw must map names to scalars")
    images = raw.get("context_images", CONTEXT_IMAGES)
    if (not isinstance(images, Mapping) or not set(_IMAGE_WINDOW) <= set(images) <= set(CONTEXT_IMAGES)
            or not all(type(images[key]) is int and 1 <= images[key] <= 50 for key in images)):
        raise RunSpecError("play.context_images takes observations and tool_results, each 1..50, and optionally "
                           "prune_at")
    if "prune_at" in images and images["prune_at"] < max(images[key] for key in _IMAGE_WINDOW):
        raise RunSpecError("play.context_images.prune_at is at least observations and tool_results (up to 50)")
    setup = raw.get("cell_setup", False)
    if type(setup) is not bool:
        raise RunSpecError("play.cell_setup is true or false")
    if setup and (runtime.get("kind") != "yam" or runtime.get("rig_mode") != "on"):
        raise RunSpecError("play.cell_setup runs a real cell's setup first: YAM on rig_mode on only")
    # ``play.code_mode`` (``ora play --code-mode``): the agent works in ORA's code mode (bash and the robot library
    # in ORA's sandbox, the environment baseline's ``policy.code_mode``) instead of Pi's own tools and codemode.
    code_mode = raw.get("code_mode", False)
    if type(code_mode) is not bool:
        raise RunSpecError("play.code_mode is true or false")
    if code_mode and not policy.get("code_mode"):
        raise RunSpecError("play.code_mode needs policy.code_mode: the environment's baseline has no code mode "
                           "to default to (RoboDojo's has), so name one in policy.code_mode")
    if policy.get("code_mode") and not code_mode:
        raise RunSpecError("policy.code_mode in free play needs play.code_mode: true (ora play --code-mode)")
    priority = request_priority(raw.get("request_priority"))
    # ``play.pipeline_predict`` (``ora play --pipeline-predict``): the next step is predicted while a motion runs and
    # used only when the receipt confirms the motion (ora-pi/src/play/pipeline.ts). Pi's own agent only.
    predict = raw.get("pipeline_predict", False)
    if type(predict) is not bool:
        raise RunSpecError("play.pipeline_predict is true or false")
    if predict and code_mode:
        raise RunSpecError("play.pipeline_predict predicts Pi's next tool call; ORA's code mode has none to predict")
    return {"schema": PLAY_SCHEMA, "scene": scene, "scenes": normalized, "instruction_budgets": bounded,
            "context_images": dict(images), "draw": dict(draw), **({"cell_setup": True} if setup else {}),
            **({"code_mode": True} if code_mode else {}),
            **({"request_priority": priority} if priority is not None else {}),
            **({"pipeline_predict": True} if predict else {})}


#: The request kinds ``play.request_priority`` names, and the bound of each value.
REQUEST_PRIORITY_KINDS = ("correction", "default")
REQUEST_PRIORITY_BOUND = 1000


def request_priority(value: Any) -> dict[str, int] | None:
    """``play.request_priority`` (``ora play --request-priority``), validated; None when absent.

    The ``priority`` field each model request then carries, for a self-hosted server's priority
    scheduling: ``correction`` on the call that first reads an operator message joined to the running
    instruction, ``default`` on every other call. vLLM runs lower values first and refuses the field
    unless started with ``--scheduling-policy priority``; SGLang runs higher values first unless
    started with ``--schedule-low-priority-values-first``. Absent, requests carry no ``priority``.
    """

    if value is None:
        return None
    if (not isinstance(value, Mapping) or set(value) != set(REQUEST_PRIORITY_KINDS)
            or not all(type(value[key]) is int and abs(value[key]) <= REQUEST_PRIORITY_BOUND
                       for key in REQUEST_PRIORITY_KINDS)):
        raise RunSpecError("play.request_priority takes correction and default, each an integer in "
                           f"-{REQUEST_PRIORITY_BOUND}..{REQUEST_PRIORITY_BOUND}")
    return {key: int(value[key]) for key in REQUEST_PRIORITY_KINDS}


def _require_backend_scenes(runtime: Mapping[str, Any], scenes: list[dict[str, Any]]) -> None:
    """Scenes a backend can reset to in the same worker (episode.play.PlayLoop._reset_scene).

    The rule is the backend's play hook's: RoboDojo relaunches on seeds of one
    scene of a scene set, RoboTwin resets within one scene configuration and
    scene set, LIBERO to an initial state below 50.
    """

    selected = find(runtime.get("kind"))
    if selected is not None and selected.play is not None:
        selected.play.require_scenes(runtime, scenes)


def bind_play_runtime(runtime: Mapping[str, Any], play: Mapping[str, Any] | None) -> dict[str, Any]:
    """The plan runtime: a play scene's backend binding, which withholds the official predicate.

    ``play.binding.bind_play_runtime`` selects it and refuses it to a benchmark
    plan (``play`` None), whose runtime it returns unchanged.
    """

    if play is None:
        return binding.bind_play_runtime(runtime, scene=None)
    return binding.bind_play_runtime(runtime, scene=play["scene"], scenes=play.get("scenes", [play["scene"]]))


def verify_play_qualification(play: Mapping[str, Any] | None, qualification: Mapping[str, Any]) -> None:
    """A RoboTwin or RoboDojo play scene's record is the one the plan binds (``play.binding``)."""

    if play is not None:
        binding.verify_play_qualification(play, qualification)


__all__ = ["PLAY_EXPERIMENT", "PLAY_SCHEMA", "REQUEST_PRIORITY_KINDS", "bind_play_runtime", "request_priority",
           "resolve_play", "verify_play_qualification"]
