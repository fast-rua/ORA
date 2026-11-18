# observe

Current annotated cameras and retained raw public camera images.

`tool.run(api, args)` runs on every toolkit-only surface and under `ora toolkit replay --tool observe`.

Current views draw measured L/R controlled points and selected solved marks.
Historical views keep original PNG bytes and select only verified public-state
steps in the current round. Pixels use top-left (u,v); positions use world metres.
The tool preserves public facts, image mapping and measurement-contract fields.
A surface declaring `two_view_pointcloud_v1` adds `pointcloud_top`, `pointcloud_front` and
`pointcloud_side` (`_pointcloud.py`, `lib/pointcloud.py`): orthographic views of the agentview
and wrist RGB-D fused inside the workspace, recomputed per call, carrying only `history_point_ids`
marks. They are derived, so they have no calibration in `public_robot_facts` and no saved step.

Replay consumes the exact recorded publication request; overlays are pure
lib.images computation. Selecting a different historical observation needs its
recorded query, otherwise NotRecorded. Public facts are supplied by the trusted
public-facts projector, with no access to truth. Per-camera integrity refusals
are retained so a missing unrequested camera does not block another view.
Derived overlay publication validates immutable bytes at the existing live
location; replay stores only evaluation CAS artifacts.

Imports are limited to toolkit lib/interface/tools and standard numeric Python.
No arbitrary paths or network calls. The static boundary check is not an OS
sandbox; tools still execute in the same process.
