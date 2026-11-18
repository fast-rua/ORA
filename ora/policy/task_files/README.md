# Task files

A RunSpec may name its own task file here (`policy.task_file`,
`ora/policy/task_files/<name>.md`). It takes the place of the reviewed
completion contract in the episode-task layer: the reviewed contract is then
not delivered, the plan's `policy.task_contract` records
`reason: replaced_by_task_file` with the file's path and sha256, and the run is
not limited to the layouts a contract record was reviewed on (RoboDojo: any
official layout of evaluation seeds 0-2, instruction read back from the reset).

Task files are task knowledge their author maintains, such as an RSI agent's
completion conditions and experience; no review vouches for them. They usually
arrive through a source overlay (`docs/reference/source-overlay.md`), which may
replace files in this directory.
