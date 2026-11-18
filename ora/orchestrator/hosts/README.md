# Execution host profiles

Two generic profiles ship here: `local-cpu` (`ora local`, CPU MuJoCo LIBERO)
and `single-gpu` (`ora setup` / `ora run`: one task on one CUDA GPU, a slot per device index 0-7, `ora run --gpu N`). For any other host,
write `<host-id>.yaml` from the template `examples/hosts/example-host.yaml` and
point `ORA_HOSTS_DIR` at its directory; a profile there with the same id wins
over the shipped one.
