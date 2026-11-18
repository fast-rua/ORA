# rotate_wrist

Rotate about the embodiment-declared approach axis in end-effector coordinates,
keeping the measured world position. Public input is degrees. Uses the shared
pose compiler, public clearance/IK references and trusted synchronous executor.
Replay consumes recorded state/queries/actions; missing evidence fails.
