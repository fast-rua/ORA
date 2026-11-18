"""A plan's context pack, served to the Pi core's view (``context_pack.*`` RPCs).

The TypeScript core applies the pack after its Context projection
(``ora-pi/src/context/context_pack.ts``); what needs the plan, the pack's
files or Pillow comes from here:

``context_pack.delivery``
    The static messages the pack replaces, each named by the sha256 of its
    exact text in the session's history, with its replacement (``null``
    drops it), and the advertised tools with the pack's description overlay.
    Every call checks the pack's files against the plan again.
``context_pack.images``
    The compact view's kept images, downscaled to ``image_long_edge_px``
    (``ora.agent.composition.prompt.context_pack.downscale_data_url``).

A plan without ``policy.context_pack`` registers neither method.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ora.agent.composition.prompt.context_pack import (
    canonical_sha256, downscale_data_url, load_bound_pack, load_tool_overlay, overlay_tools,
    render_episode_task, sha256_text,
)
from ora.agent.contracts import PolicySessionError


def context_pack_handlers(planner: Any) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    binding = planner.policy.get("context_pack")
    if not isinstance(binding, Mapping):
        return {}
    profile = binding["profile"]

    def delivery(_params: Mapping[str, Any]) -> dict[str, Any]:
        task = planner.bound_task_instruction
        if task is None:
            raise PolicySessionError("context pack delivery was asked for before the task was bound")
        pack = load_bound_pack(binding)
        messages: list[dict[str, Any]] = []
        if "system" in binding:
            messages.append({"role": "system", "content_sha256": sha256_text(planner.harness_system_template),
                             "content": binding["system"]["text"]})
            messages.append({"role": "user", "content_sha256": sha256_text(planner.project_template),
                             "content": None})
        if "episode_task" in binding:
            messages.append({
                "role": "user",
                "content_sha256": sha256_text(render_episode_task(planner.episode_prefix_template, task)),
                "content": render_episode_task(binding["episode_task"]["template"], task),
            })
        tools = None
        overlay_file = (profile.get("tools") or {}).get("overlay_file")
        if overlay_file is not None:
            tools = overlay_tools(planner.gateway.definitions(), load_tool_overlay(pack, overlay_file))
            if canonical_sha256(tools) != binding.get("delivered_tools_sha256"):
                raise PolicySessionError("the context pack's overlaid tools differ from the planned ones")
        return {"messages": messages, "tools": tools}

    def images(params: Mapping[str, Any]) -> dict[str, Any]:
        long_edge = profile["compact_view"]["image_long_edge_px"]
        urls = params.get("images")
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise PolicySessionError("context_pack.images takes a list of data URLs")
        return {"images": [url if long_edge is None else downscale_data_url(url, long_edge) for url in urls]}

    handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {"context_pack.delivery": delivery}
    if "compact_view" in profile:
        handlers["context_pack.images"] = images
    return handlers


__all__ = ["context_pack_handlers"]
