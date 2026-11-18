# Round Static Facts

What stays fixed within a round -- a fixed camera's calibration, the static robot facts -- is stated once per round, in the round_static_facts message that follows the round's first observation; an observation leaves out each current_actual field whose current value is byte-identical to that message, carries every changed value inline, such as a moving camera's changed pose, and lists under static_facts.absent any field of that message that no longer has a current value.
