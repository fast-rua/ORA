"""Public receipt projection for default tools; keep controller evidence internal."""


def model_receipt(receipt):
    """Return minimal_action_effect plus spend/identity, without stage-only evidence."""
    return {key: value for key, value in receipt.items() if key != "stage_evidence"}


def model_stages(stages):
    """Preserve stage outcomes while projecting each embedded receipt."""
    return [{**stage, "receipt": model_receipt(stage["receipt"])} for stage in stages]
