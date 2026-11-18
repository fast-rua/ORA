"""The reviewed RoboDojo Assets revision of a card; host-side, no simulator imports.

User decision 2026-11-14: every task reads one Assets checkout, at
``REVISIONS["assets"]`` (1127bda, pinned with RoboDojo f57e163). It serves the
regenerated swap_T layouts the previous pin a14409d7 no longer served, so the
swap_T-only successor checkout (35efbc7d, ``ROBODOJO_SWAP_T_ASSET_ROOT``) is
retired. From 35efbc7d to 1127bda only
``Object/RoboDojo/Rigid/mug/00014/metadata.json`` changed, a mug that only
``hang_mugs_random`` layouts hold. Records, cards and installations made at the
older revisions stay as they were written; none is reused at this pin.
"""
from __future__ import annotations


def unchanged_card_assets(original: dict, current: dict, *, measured_task: str) -> bool:
    """A card is reused only on exactly the source revisions it was measured on.

    The one reviewed cross-revision transition (a14409d7 cards reused by a
    35efbc7d swap_T lane) ended with the single asset revision; whatever task
    the card measured, its revisions must equal the runtime's.
    """
    return original == current
