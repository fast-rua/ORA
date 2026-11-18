"""Write one explicit issue through the episode-control capability."""
from ...lib.control import report_issue
from ...lib.values import JsonValue


def run(api, arguments):
    """Replay: compare the recorded issue request; never write the source run.

    Live validates and appends a public issue with the trusted episode writer.
    This control operation spends no native actions and returns the released
    success/error envelope; it does not claim official task success.
    """
    return report_issue(api, JsonValue.of(arguments)).data.json()
