"""Parameterized rigid jaw-frame compilation, separate from candidate selection."""
from .values import JsonValue
from . import _grasp_frames


def compile_contact(candidate: JsonValue, compiler: JsonValue, *, public_contact_pose: bool = True) -> JsonValue:
    """Pure computation: compile a source jaw frame into a named controlled-point pose.

    Example: compile_contact(raw_candidate, public_compiler_identity). Translations
    and openings are metres; rotations are xyzw unit quaternions. The identity
    supplies the source-origin offset, approach/opening limits, and axis mapping.
    Returns every compiler check and pose; invalid geometry raises ValueError.
    This is the general X5/ALOHA mapping, or the grip-site handoff of the Panda and
    the Robotiq TCP, which keeps its own limits and check names and always carries
    the contact pose.
    No IK, collision or contact success is inferred. Candidate ranking is a tool choice.
    """
    if (compiler.json().get('controlled_point') not in {'end_effector', 'arx_x5_link6', 'link6'}
            and compiler.json().get('id') not in _grasp_frames.GRIP_SITE_STYLE_COMPILER_IDS):
        raise ValueError('compiler requires a controlled-point mapping this library compiles')
    return JsonValue.of(_grasp_frames._compile_grasp_candidate(
        candidate.json(), compiler_identity=compiler.json(), public_contact_pose=public_contact_pose))
