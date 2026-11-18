# rotate_pitch

Rotate about local Y in end-effector coordinates, keeping measured world
position. Public input is degrees. Uses the shared pose compiler, public
clearance/IK references and trusted synchronous executor. Replay consumes
recorded state/queries/actions; missing evidence fails.
