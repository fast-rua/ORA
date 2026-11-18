# Native action normalization regression

Public action envelopes extracted from the successful arrange_largest_number
layout-0 run identified in move-pose-normalization.json, source revision
2c05978ec3e2a2478b05f6a9f85b7263a593e46c. Parent tool call 14, native receipt
11, action-6297883f2a36bf62, first tick 37. JSON field values are preserved;
formatting is canonicalized for the fixture. No observations, GT or service
outputs are copied. The interface stores the submitted quaternion, while the
trusted execution ledger stores validate_runtime_action(request). Tests compare
this exact owner form and reject other field differences without tolerance.
