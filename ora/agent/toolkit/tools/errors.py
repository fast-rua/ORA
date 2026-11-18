"""Portable tool refusal; trusted live assembly preserves the existing gateway envelope."""


class ToolRefusal(RuntimeError):
    """A refusal with explicit public fields, independent of the live coordinator.

    ``images_base64`` (PNG, base64) travel beside the fields, never in them: a picture of what was refused that the
    gateway attaches as the result's images (``_dry_run.pictured``)."""

    images_base64 = ()

    def __init__(
        self,
        message,
        *,
        code="invalid_arguments",
        retryable=True,
        field=None,
        received=None,
        constraints=None,
        retry_requires_changed_arguments=None,
    ):
        super().__init__(message)
        self.fields = dict(
            code=code,
            retryable=retryable,
            field=field,
            received=received,
            constraints=dict(constraints or {}),
            retry_requires_changed_arguments=retry_requires_changed_arguments,
        )

    @classmethod
    def from_public(cls, error):
        return cls(
            error["message"],
            **{
                key: value
                for key, value in error.items()
                if key not in {"message", "infrastructure"}
            },
        )

    @property
    def code(self):
        return self.fields["code"]

    @property
    def constraints(self):
        return self.fields["constraints"]
