# Compute usage v2 with routing v1

Compute evaluates numbers already supplied by the policy. It does not obtain
new observations or execute robot commands. Routing v1 requires a call before
using a newly calculated number to choose or parameterize an action, including
a single offset or angle. It preserves direct copying of valid tool values,
reuse while inputs and assumptions are unchanged, and selection of constants
that need no calculation. Image, depth and calibration queries belong to
`geometry_query`; named current-pixel measurements belong to `mark_point`.
The exact model-visible description on YAM is its file-backed surface's
[`yam_direct_v1/compute.json`](../../ora/policy/tool_surfaces/yam_direct_v1/compute.json),
and on LIBERO, RoboTwin, RoboCasa, RoboLab and RoboDojo the tool's own
[`compute/description.json`](../../ora/agent/toolkit/tools/compute/description.json).
The evaluator, input schema and output
schema retain their v2 behavior. The defaults state the rule once, in
`compute`'s Use when, and the pose tools point to it
([reference](../reference/public-interfaces.md#tool-description-consistency)).

## Prebound names

No import statement is needed: the evaluator already provides `np`, `Rotation`,
`math`, `geom`, `pi`, `float` and `int`. These are restricted numeric facades;
the full NumPy/SciPy/math module environment is not exposed. Inputs in `bindings`
are automatically available as variables; numeric lists become arrays.

| Operation | Expression |
|---|---|
| Distance between points | `np.linalg.norm(b - a)` |
| Degrees to radians | `math.radians(angle)` |
| Normalize a nonzero direction | `geom.unit(v)` |
| Unsigned angle, degrees | `geom.angle_deg(a, b)` |
| Rotate a vector, quaternion xyzw | `geom.rotate(q, v)` |
| Transform a point | `geom.transform_point(t, q, p)` |
| Euler angles to quaternion | `Rotation.from_euler('xyz', angles, degrees=True).as_quat()` |

The point transform is `t + Rotation(q).apply(p)`. The caller supplies frame
identities and consistent units. Normalization scales before taking a norm to
avoid overflow/underflow; a zero vector is rejected because it has no direction.
No helper turns an assumed geometric relationship into a measured one.

## Complete call

```json
{
  "bindings": [
    {"name": "a", "value": [0.1, 0.2, 0.3]},
    {"name": "b", "value": [0.4, 0.6, 0.3]}
  ],
  "expressions": "distance_m = np.linalg.norm(b - a)",
  "outputs": ["distance_m"]
}
```

The numeric result is `distance_m = 0.5`. Replace these illustrative values with
actual inputs. Optional `from` on a binding records a source label; it does not
verify that source. Each call starts fresh, so pass inputs again on later calls.

`expressions` is one JSON string, with line breaks encoded as `\n`. Do not put
Markdown fences, import, def, print or return statements inside it. `outputs`
selects final variable names; a final bare expression is named `result`.
Omitting the selection retains the existing automatic numeric-variable output.
After JSON decoding, statement separators must be real newlines or semicolons;
literal backslash-n characters between Python statements cause a syntax error.
Correct that source and retry instead of substituting a mental answer.

## Limit errors

- Output count/size: select only the final values. A temporary array may be
  larger than a returned value when it stays within execution memory bounds;
  unselected values are not serialized. Alternatively return a slice or summary.
- Unknown name/import: use a prebound name or supply the value in `bindings`.
- Non-finite number: check zero division, magnitude and units. For vector
  directions and angles, prefer the scaled helpers.
- Source/time/memory limit: shorten the expression with the prebound operators
  or split independent calculations. Repeating identical input does not help.

The unchanged bounds are 4096 source bytes, 24 statements, 16 published input
bindings, 12 returned values, 64 elements and two dimensions per returned value,
512 MiB child data limit, two CPU seconds and five seconds for the child-result
wait. Selection does not bypass these bounds for a returned value. No arbitrary
imports, filesystem/network access or persistent notebook state are available.

## Validation and activation

The two model-visible JSON examples are part of the `compute` description
itself (the layers named above). `tests/ora/agent/toolkit/host/compute/test_compute_tool.py`
checks that each appears verbatim in the delivered description of every
applicable surface and executes it through that surface's real schema and
handler. Helper math, output selection and isolation tests run offline in the
actual evaluator worker. These checks prove delivery/execution contracts, not
fewer model errors or better task success. Historical runs retain their
original source.

Every default surface (LIBERO-Pro, RoboTwin, RoboDojo, RoboCasa365 and
RoboLab) declares `compute` with the current routing generation, the Compute v2
helpers and output selection; `./scripts/ora baselines --explain` lists the
resolved tools.
