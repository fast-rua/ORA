## How this context reaches you

The conversation grows by appending: what you were shown stays where it is, in
the form you first received it, until the history gets long.

Camera images of each observation stay in the message where the observation
arrived, each after its label (the sequence and the camera), so the images of
your latest turns sit in the history beside what you did. Compare an
observation's images with an earlier one's when you need to see what an action
changed.

What you wrote stays as you wrote it: your messages and the arguments of
every tool call (a script, a file's content) are never shortened.

When the history gets long, the data in every turn older than your latest few
is shortened in one step, and the history then grows by appending again; while
it stays long, the step repeats every few turns. The turn of your latest robot
action -- its call, its result and the observation after it -- stays as it was
however old. In those older turns:

- a tool result longer than 8,192 bytes keeps only its status fields (such as
  `ok`, `status`, `exit_code` and the actions it executed, and for a failed one
  the `error` type and message), its size in `result_bytes`,
  `"externalized": true` and a `recoverable_artifact`; a shorter one stays as
  it was;
- an image a tool returned, and each camera image of an observation, is
  replaced by its `<recoverable_artifact>` line, whose `open` is observe:
  observe with that `ref` shows the image again, and observe with an
  observation's step shows saved views of the current round again, as they
  were then;
- an older observation keeps its identity, with `"observation_externalized":
  true` and a `recoverable_observation`, in place of its measurements.

Nothing is lost: artifact_read with the reference's `ref` returns the
original text, and `json_pointer` selects one field of it (for example
`/output`). Read one back only when you need it; a file you wrote is usually
quicker to read from where you saved it. These references are the history's,
never arguments: a tool call always takes its own real arguments.

If even that is not enough, every turn before your latest few is replaced by
one checkpoint summary, which names the exact transcript it summarizes; the
turn of your latest robot action stays. The images of the summarized turns are
not kept. Evidence you can no longer see has not been withdrawn, contradicted
or made false.

What stays fixed within a round -- a fixed camera's calibration, the static
robot facts -- is stated once per round, in the round_static_facts message that
follows the round's first observation. An observation leaves out each
current_actual field whose current value is byte-identical to that message,
carries every changed value inline, such as a moving camera's changed pose, and
lists under static_facts.absent any field of that message that no longer has a
current value. An observation states terminal_lifecycle only when it is true.
