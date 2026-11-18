"""Read current state through its owner; never capture a value at assembly."""
from dataclasses import dataclass
from typing import Any
from ..types import AccessDenied, NotRecorded, Response


@dataclass(frozen=True, slots=True)
class State:
    _backend: Any
    _access: Any
    _check_task: bool = False

    def read(self, key: str) -> Any:
        """Live/replay: read the selected state's current value. Privileges use separate capabilities."""
        if key.startswith('privileged.'):
            raise AccessDenied('privileged state is unavailable through state.read')
        from ...state.keys import StateError, StateAccessDenied
        try:
            return self._access.read(key)
        except StateAccessDenied:
            raise
        except StateError as error:
            raise NotRecorded(f"未记录: {key}") from error

    def write(self, key: str, value: Any) -> None:
        """Live: write finite JSON to tool.*. Loaded replay snapshots are read-only."""
        if not key.startswith('tool.'):
            raise AccessDenied('tools may write only tool.*')
        self._access.write(key, value)

    def observation(self, *, observation: dict | None = None) -> Response:
        """Read the currently selected publication, including original file references."""
        return self._backend.observation(observation=observation)

    def proprioception(self, *, observation: dict | None = None) -> Response:
        """Read measured joint radians, normalized gripper and episode-local pose metres/xyzw."""
        return self._backend.proprioception(observation=observation)

    def receipt(self, *, action_id: str) -> Response:
        """Read a completed receipt at this boundary; never a future result."""
        return self._backend.receipt(action_id=action_id)

    def check_task(self) -> Response:
        """Transitional single-key grant; no general privileged capability is exposed."""
        if not self._check_task:
            raise AccessDenied('check_task capability disabled')
        return self._backend.check_task()
