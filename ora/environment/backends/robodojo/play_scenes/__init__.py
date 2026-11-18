"""Worker-only modules of ORA's own RoboDojo free-play scenes (``ora play --env robodojo``).

``scenes/<set>/<scene>.yml`` are the scene YAMLs and ``scenes/<set>/index.yml``
each set's scene entries (label, physics device, settle steps, assets,
description); the deterministic per-seed
layout generator is host-safe pure Python
(``ora/environment/benchmarks/robodojo_play_layouts.py``). ``lifecycle.py`` (the
reward-free TaskEnv lifecycle), ``bootstrap.py`` (the play twin of the
benchmark Kit boot), ``session.py`` (the ``robodojo_play`` worker session) and
``audit.py`` (the play-scene audit worker) import the RoboDojo worker backend
and, inside their functions, Isaac Sim; they run only in the isolated RoboDojo
interpreter. The host-safe registry of the sets is
``ora/environment/benchmarks/robodojo_play_scenes.py``.

None of these modules is in the benchmark worker kernel
(``ora/orchestrator/robodojo_kernel.WORKER_KERNEL_PATHS``): ``environment/
worker.py`` reaches ``session.py`` only on its ``robodojo_play`` branch, a
pruned dispatch edge (``robodojo_kernel_review.DISPATCH_PRUNED_EDGES``). They
have their own code identity, the play kernel
(``ora/orchestrator/robodojo_play_kernel.py``), which play-scene audits bind
together with each scene's own source binding (its index entry, its YAML and
the generator: ``robodojo_play_scenes.source_binding``).
"""
