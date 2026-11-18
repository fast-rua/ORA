"""The closed-skill stage driver and receipt.

Both delegate to ``toolkit/host/skills/driver``. The toolkit serves
``align_plan`` and the closed skills (``dispatch.install``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

from typing import Any


class SkillRuntimeMixin:
    """The closed-skill stage driver and receipt."""

    def drive_skill_stage(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ora.agent.toolkit.host.skills import driver as skill_driver
        return skill_driver.drive_skill_stage(self, *args, **kwargs)

    def build_skill_receipt(self, **kwargs: Any) -> dict[str, Any]:
        from ora.agent.toolkit.host.skills import driver as skill_driver
        return skill_driver.build_skill_receipt(self, **kwargs)
