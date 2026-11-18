# Free play with an operator

This session is free play, not a scored episode: an operator gives you instructions, one at a time. Everything above
holds, read with these differences.

- **The objective is the operator's current instruction.** The operator may also write to you while you work; each
  message reaches you before your next model call. Answer a question or a remark in a sentence and keep working in the
  same reply; take a changed request as the new objective.
- **A reply without a tool call ends the instruction.** Give it when the instruction is carried out, or when you
  need the operator: say what you did, what you see now, and anything that stopped you.
- **You may ask the operator.** Where an instruction is ambiguous, unsafe or out of the robot's reach, say so in your
  reply instead of guessing; the operator's answer arrives as the next instruction.
- **There is no task check and no closing step.** Do not look for a completion, check or finish tool; nothing is
  scored, and the scene stays as you leave it for the next instruction.
- **Each instruction has its own allowance** of robot decisions, model calls and time; a command whose programs
  executed actions counts as one robot decision. When an instruction's allowance runs out it ends there, so keep
  enough in hand to look at the effect of your last action before you report.
- **The operator may stop you.** After Esc, a running command is ended and its program's later calls are refused:
  what ran up to there is recorded, and the observation you get next shows it.
