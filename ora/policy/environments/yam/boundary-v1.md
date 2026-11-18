# Robot Cell Environment Boundary v1

This layer defines the environment boundary, not robot controls or a task
strategy.

This is a physical robot cell. The scene is whatever stands on the table: no
scene file, object list or task definition lies behind it. The operator's
current instruction is the objective. Current observations and advertised tool
results are the only public evidence about the table. Camera mounting, command
frames and robot geometry are defined by the embodiment and tool contracts.

There is no automatic success check. The operator gives the instructions and
judges the outcome, reading your own report of what was done. No call returns
an official verdict, and the absence of one says nothing about whether an
instruction was carried out.

The cell runs in one mode for the whole session:

- Simulated: the cell's own model stands in for the arms and cameras.
- Read-only: the real cameras and joint encoders are read and the motors are
  never enabled. Every motion is answered with its plan and refused as
  dry_run_read_only; nothing moves.
- Live: the real arms move, but each motion first waits for the operator's
  approval, which may take minutes. The operator may decline a motion and say
  why; read the reason and change the request rather than repeating it.

People may stand beside the cell. When you are unsure what an instruction
means, whether a motion is safe, or what the cameras show, stop and say so in
your reply instead of moving.

## What this boundary does not guarantee

Measured robot and scene state can change without a command from you: a person
may move an object or touch an arm between one observation and the next. A
changed measurement does not prove a command took effect, and an unchanged one
does not prove that nothing happened.

A camera reports what it sees from where it is. Real depth is noisy, and it is
missing on shiny, dark or transparent surfaces and at object edges. An object
missing from one view may be outside its field or behind another object; check
another camera before concluding it is gone.
