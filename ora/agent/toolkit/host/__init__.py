"""The trusted host the toolkit's tools run under, and the tool runtime frame they run in.

``runtime`` holds that frame: ``ManipulationToolRuntime`` and ``RobotToolRuntime``, the factory that builds one per
episode (``runtime.factory.build_tool_runtime``), the dispatch that installs the toolkit's tools on it
(``runtime.dispatch``), the artifact publisher (``runtime.tool_data``) and the input freezer (``runtime.inputs``).
The modules at this level bind a call to the toolkit: ``adapter``, ``observe``, ``geometry``, ``segment``, ``motion``,
``lifecycle``, ``robot`` and the rest publish public data and configuration to the tools, ``recording`` journals their
I/O, ``action_preview`` answers dry runs on environment-planned backends and ``observe_labels`` names a dual-arm
observe's images. The other sub-packages hold what those need: ``compute`` (the restricted geometry evaluator, whose
worker runs in a process of its own), ``finish`` (the ``finish_episode`` and ``check_task`` rules), ``grasp``,
``perception``, ``reach``, ``motion_rules``, ``skills`` (``grasp_execute``'s stage loop) and ``shared``.

The host imports the rest of the toolkit and ``composition`` (structure rule L6); ``tools/``, ``lib/`` and
``interface/`` never import the host (``tests/ora/agent/toolkit/tools/test_boundary.py``).
"""
