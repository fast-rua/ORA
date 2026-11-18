"""Trusted field projection of the original public tool refusals for new tools."""

from ora.agent.toolkit.interface.types import PublicToolFailure
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.shared import service_refusals


def service_adapter(adapter, code, *, target_ref=None):
    """Preserve the existing public message/code/constraints, not exception internals."""

    def invoke(parameters, inputs, store):
        try:
            return adapter(parameters, inputs, store)
        except ToolError as error:
            raise PublicToolFailure(error.as_dict()) from error
        except ValueError as error:
            projected = service_refusals.refusal(
                error, code=code, target_ref=target_ref
            )
            raise PublicToolFailure(projected.as_dict()) from error

    return invoke


def invoke(function, api, arguments):
    """The existing gateway recognizes ToolError; tools do not import that gateway."""
    try:
        return function(api, arguments)
    except ToolRefusal as error:
        raise ToolError(str(error), **error.fields, images_base64=error.images_base64) from error
    except PublicToolFailure as error:
        projected = ToolRefusal.from_public(error.public())
        raise ToolError(str(projected), **projected.fields) from error


def control_adapter(adapter):
    """Preserve the gateway's public control/read error projection in recorded responses."""
    from ora.agent.toolkit.interface.types import AccessDenied, NotRecorded

    def call(parameters, inputs, store):
        try:
            return adapter(parameters, inputs, store)
        except (PublicToolFailure, AccessDenied, NotRecorded):
            raise
        except Exception as error:
            projected = (
                error
                if isinstance(error, ToolError)
                else ToolError(str(error), code="internal_tool_error", retryable=False)
            )
            raise PublicToolFailure(projected.as_dict()) from error

    return call
