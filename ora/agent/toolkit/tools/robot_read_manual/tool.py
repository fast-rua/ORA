"""``robot.read_manual``: one named section of the embodiment manual."""

from ..errors import ToolRefusal


def run(api, arguments):
    """Replay: the recorded manual text; pure section selection. Section names match exactly."""
    if set(arguments) != {"section"} or not isinstance(arguments.get("section"), str):
        raise ToolRefusal("robot.read_manual requires section")
    manual = api.artifacts.tool_context()["robot_read_manual"]
    requested = arguments["section"]
    if requested not in manual["sections"]:
        raise ToolRefusal("unknown manual section", code="unknown_section", field="section", received=requested,
                          constraints={"sections": manual["sections"]})
    lines = manual["text"].splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("## ") and line[3:].strip() == requested)
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")), len(lines))
    return {"section": requested, "content": "\n".join(lines[start:end]).strip(), "manual_sha256": manual["sha256"]}
