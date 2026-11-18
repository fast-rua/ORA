"""Launch preflight: the fail-closed host, provider, simulator and qualification gates.

``gates.run_preflight`` checks one RunPlan before launch and returns the
check records (``checks``); a backend's own checks come through its
``preflight_runtime`` hook in ``backends/registry.py``. Beside it: execution
resources and host profiles (``resources``, ``resource_inventory``,
``resource_admission``), host and output storage (``host_storage``) and the
provider probe (``provider_probe_stages``, ``provider_probe_diagnostics``,
``provider_image_answer``, ``provider_gateway``).
"""
