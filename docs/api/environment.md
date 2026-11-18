# Environment sessions

The host-safe boundary exposes public observations and supervised execution.
RoboTwin, LIBERO / LIBERO-Pro, RoboDojo, RoboCasa365 and RoboLab implement it
in isolated simulator workers.

```python
from ora.environment.contracts import EnvironmentSession
from ora.environment.session import DescribedEnvironmentSession
```

::: ora.environment.contracts.EnvironmentSession

::: ora.environment.session.DescribedEnvironmentSession

See [backend authoring](../developer/add-backend.md) for observation privacy,
descriptor registration, action dispatch, and official-verdict requirements.
