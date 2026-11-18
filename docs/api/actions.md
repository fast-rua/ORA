# Action contracts

Validate an action against its versioned contract before admission. A valid
envelope still needs Supervisor checks and an executing environment. Target-state
v4 stages a combined request as pose-first, conditional-gripper phases.

```python
from ora.contracts.action import (
    ActionValidationError,
    validate_runtime_action,
    target_state_phases,
)
```

::: ora.contracts.action.ActionValidationError

::: ora.contracts.action.validate_runtime_action

::: ora.contracts.action.target_state_phases

A motion v3 `move_pose` may carry an optional `speed`: the end-effector speed
the move asks for (a preset, `speed_m_s`, `slow_approaching`). A call that
names none carries no `speed` and runs at the default; see
[Motion speed and wrist checks](../reference/motion-speed.md).

```python
from ora.contracts.motion_speed import speed_request, validate_speed
```

::: ora.contracts.motion_speed.speed_request

::: ora.contracts.motion_speed.validate_speed

[Runtime lifecycle and authority](../reference/runtime-lifecycle.md)
