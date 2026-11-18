"""CLI for trying supplied toolkit code at a recorded call boundary."""


def register(subparsers):
    parser = subparsers.add_parser("toolkit", help="try toolkit code using recorded public data")
    commands = parser.add_subparsers(dest="toolkit_command", required=True)
    from ora.app.toolkit_provider_replay import register as register_provider_replay
    register_provider_replay(commands)
    replay = commands.add_parser("replay", help="run a script before a recorded parent tool call")
    replay.add_argument("run", help="source run directory (read-only)")
    boundary = replay.add_mutually_exclusive_group()
    boundary.add_argument("--at", help="parent tool_call_id or zero-based call ordinal; legacy diagnostic unless --snapshot")
    boundary.add_argument("--step", type=int, help="completed policy-state step; implies snapshot replay, no simulator/actions")
    boundary.add_argument("--native-action", type=int, help="completed native tick count N: state at end of controller native_action=N; N=0 is reset/before any native action; requires --resource-profile; implies snapshot, forbids new actions")
    replay.add_argument("--snapshot", action="store_true", help="phase-one snapshot path: original bytes and private tool state; planner requires simulator")
    modes = replay.add_mutually_exclusive_group(required=True)
    modes.add_argument("--list", action="store_true", help="list parent calls and recording/gap availability")
    modes.add_argument("--script", help="Python file defining run(api, args)")
    modes.add_argument("--tool", help="run a migrated tool from the source surface, or check_task, at the selected boundary")
    replay.add_argument("--args", help="JSON object; required for a different tool, otherwise defaults to original arguments")
    replay.add_argument("--service-report", help="qualified perception report, as for spec robodojo compose")
    replay.add_argument("--output", help="new output directory; default workspace/toolkit-replay/run_id/UTC-random outside runs")
    replay.add_argument("--live", action="store_true", help="restore a simulator on an already allocated resource profile")
    replay.add_argument("--resource-profile", help="caller-allocated live simulator lane")
    replay.add_argument("--execute-action", action="store_true", help="execute one live proposal after the tool returns")
    replay.add_argument("--json", action="store_true", help="print machine-readable list or result including output path")


def run(args):
    if args.toolkit_command == "provider-replay":
        from ora.orchestrator.evaluation.provider_replay import run as provider_replay
        return provider_replay(args)
    import json
    from pathlib import Path
    from ora.agent.toolkit.interface.types import decode
    from ora.orchestrator.toolkit.selection import list_calls
    from ora.orchestrator.toolkit.replay import experiment

    try:
        source = Path(args.run).resolve(strict=True)
        if args.list:
            if any((args.at, args.step is not None, args.native_action is not None, args.snapshot, args.args, args.service_report, args.output, args.live, args.resource_profile, args.execute_action)):
                raise ValueError("--list cannot be combined with execution options")
            result = list_calls(source)
        else:
            if args.snapshot or args.step is not None or args.native_action is not None:
                if args.live or args.execute_action or (args.resource_profile and args.native_action is None):
                    raise ValueError("snapshot replay cannot mix with legacy --live/--execute-action; use plain --at for legacy diagnostics")
                if args.at is None and args.step is None and args.native_action is None:
                    raise ValueError("--snapshot requires --at, --step or --native-action")
                from ora.orchestrator.toolkit.snapshot_replay import experiment as snapshot_experiment
                result = snapshot_experiment(source=source, at=args.at, step=args.step,
                                             native_action=args.native_action, resource_profile=args.resource_profile,
                                             script=None if args.script is None else Path(args.script), tool=args.tool,
                                             arguments=None if args.args is None else decode(args.args.encode()),
                                             report=None if args.service_report is None else Path(args.service_report),
                                             output=None if args.output is None else Path(args.output))
            else:
                if args.at is None:
                    raise ValueError("--at is required for replay")
                result = experiment(source=source, at=args.at,
                                    script=None if args.script is None else Path(args.script),
                                    tool=args.tool, arguments=None if args.args is None else decode(args.args.encode()),
                                    report=None if args.service_report is None else Path(args.service_report),
                                    output=None if args.output is None else Path(args.output),
                                    live=args.live, resource_profile=args.resource_profile, execute_action=args.execute_action)
    except (ValueError, OSError, KeyError, TypeError) as error:
        result = dict(status="error", error=dict(type=type(error).__name__, message=str(error)))
    if args.json or result.get("status") == "error":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.list:
        for call in result["calls"]:
            print(f"{call['call']}\t{call['tool_call_id']}\t{call['tool']}\t"
                  f"toolkit_io={call['has_toolkit_io']}\tgap={call['has_gap']}")
        print(f"unbound_gaps={result['unbound_gaps']} unknown_parent_gaps={result['unknown_parent_gaps']}")
    else:
        print(f"Replay output: {result['output']}")
        print(json.dumps(result["result"], ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "error" else 0
