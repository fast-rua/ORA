"""Bind the existing restricted numeric kernel, never a tool handler or an eval API."""

from ora.agent.toolkit.host.compute.evaluator import shared_runtime
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.interface.types import PublicToolFailure


def evaluate(source, bindings, *, outputs=None):
    try:
        return shared_runtime().evaluate(source, bindings, outputs=outputs)
    except ToolError as error:
        raise PublicToolFailure(error.as_dict()) from error
