# RoboDojo reset crash diagnostics

These operator-only environment controls support a controlled comparison of
initial-reset SIGSEGV failures. They do not change tools, prompts, model inputs,
physics, rendering, action budgets, reset arguments or startup timeouts. They
apply to the benchmark `RobodojoRoundMixin._reset_environment` path, including
round restarts/layout relaunches; they do not change the separate free-play
reset implementation or the 120-second environment-construction diagnostic.

| Environment variable | Default | Values and effect |
| --- | --- | --- |
| `ORA_ROBODOJO_RESET_TRACEBACK` | `on` | `on`: 60-second stack timer; `off`: do not arm that timer |
| `ORA_ROBODOJO_RESET_FATAL_TRACEBACK` | `off` | `on`: install an independent Python fatal-signal traceback file for the duration of reset, if no Python fatal handler already exists |

Only the exact values `on` and `off` are accepted; a typo is a configuration
error before reset, not a silent assignment to the control group. The variables
are explicitly allowlisted through the environment RPC launcher. Set them on
the allocated router task/entry process so Supervisor and worker inherit them;
no RunSpec or baseline change is required. No credentials are added to the
worker's environment.

For the timer comparison, keep fatal logging identical in both groups:

```bash
# Control: historical timer, plus opt-in fatal log shared by both groups.
export ORA_ROBODOJO_RESET_TRACEBACK=on
export ORA_ROBODOJO_RESET_FATAL_TRACEBACK=on

# Treatment: same setup, only the timer changes.
export ORA_ROBODOJO_RESET_TRACEBACK=off
export ORA_ROBODOJO_RESET_FATAL_TRACEBACK=on
```

Use the same task/layout, code, concurrency and runtime allocation policy in
both groups. Do not mix a changed concurrency cap into the timer experiment.
Default production behavior remains timer on, fatal log off. The evaluation
owner submits real runs; this feature does not allocate resources or retry them.

## Stack timers

`stack_timer` in `ora/environment/backends/robodojo/reset_diagnostics.py` writes every
thread's Python stack once, if the block outlasts its time: 60 seconds around a
benchmark reset, 120 around environment construction and free-play resets. It runs on a
Python thread holding the GIL, so a C call that keeps the GIL delays the dump, and it
never ends the run.

These timers used to be `faulthandler.dump_traceback_later`, which reads other threads'
frames without the GIL. That crashed workers: the reset comparison above was opened for
SIGSEGVs at reset, and a later batch lost 5 of 38 firings the same way, each with
`worker_returncode` -11 and a dump cut off mid-frame. Plain Python 3.11 reproduces it
when code objects and threads come and go under the dump.

## Evidence and limits

Before initial reset, the worker writes
`native/reset-initialization-stack.diagnostics.json` with schema
`ora.robodojo-reset-diagnostics.v1`. It records PID, start time, effective timer
setting/60-second duration, fatal request and effective handler, artifact names,
current soft/hard `RLIMIT_CORE` in bytes (`-1` means unlimited), the Linux
`core_pattern` when readable, and `outcome=in_progress`. On normal return it
adds finish time and `outcome=completed`; a Python exception records
`outcome=exception` and its type, then re-raises the same exception. A hard crash
leaves the pre-reset record; `in_progress` alone is not proof of a crash—check
`environment-session.json.worker_returncode` and the run result.

The timer file remains `reset-initialization-stack.txt` (empty when disabled).
The optional fatal log is `reset-initialization-stack.fatal.txt`. Round resets
use the corresponding `reset-initialization-stack-round-XXXX` stems. The
handler is disabled before its file closes; only the handler installed by this
scope is removed. If a Python fatal handler already exists, it is preserved
and evidence says `fatal_handler=existing_handler_preserved`, `fatal_file=null`.
An installation error is recorded as `setup_failed` without preventing reset.
Evidence-write I/O failures log a warning and do not mask reset's own outcome.

This is a Python fatal traceback, **not a native C/C++ backtrace or guaranteed
core dump**. A crash in a dumper/handler may still truncate it. The code does not
raise process core limits, alter `core_pattern`, or configure a node collector.
Use the recorded limits and the operator's existing core collection facility
if a native backtrace is needed. No crash report is uploaded by this feature.

The helper is included in both worker and motion source identities. Existing
qualification identities may therefore warn of drift under Q32; do not hide
that evidence. Default model-visible content is unchanged. CPU tests cover
timer defaults/off mode, reset/relaunch arguments, exception identity and file
lifetime, preservation of an existing handler, environment forwarding, and a
deliberately signaled child process with core dumping disabled. They do not
establish whether turning off the timer fixes the real simulator crashes.
