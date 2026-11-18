# Shared by the repository's shell wrappers: select the project interpreter.
#
#   ROOT=<the checkout>; . "$ROOT/scripts/lib/python.sh"
#   ora_python NAME             NAME is the wrapper, named in any notice
#   ora_python --stdlib NAME    an entry that needs only the standard library
#
# It sets PYTHON, one interpreter path, in this order:
#   1. the checkout's .venv;
#   2. an explicit $PYTHON;
#   3. uv's project environment, $UV_PROJECT_ENVIRONMENT (a shared locked
#      venv), used as it is: its bin/python runs directly, never through uv;
#   4. only when none of those is set and uv is installed: `uv run --locked
#      python`, which creates or syncs the checkout's .venv from uv.lock (a
#      fresh clone) and names its interpreter, so the next run takes step 1;
#   5. python3 from PATH, said on stderr, because any python3 on PATH (a conda
#      base, for one) runs outside the locked environment.
# A set $UV_PROJECT_ENVIRONMENT without bin/python goes to step 5, never to
# uv: uv run would create or re-sync that shared environment. So does a set
# $UV_NO_SYNC, under which uv run would create an empty .venv. A --stdlib
# entry skips step 4 (its code needs no environment) and takes step 5
# silently. The checkout goes first on PYTHONPATH, so an environment installed
# from another checkout still runs this one's code. scripts/ora keeps its own
# copy of steps 1-3 and 5 so that it also works on its own.

ora_python() {
  _ora_python_stdlib=
  if [ "${1:-}" = --stdlib ]; then
    _ora_python_stdlib=1
    shift
  fi
  _ora_python_name=${1:-ora}
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif [ -n "${PYTHON:-}" ]; then
    :
  elif [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
  elif [ -z "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -z "${UV_NO_SYNC:-}" ] && [ -z "$_ora_python_stdlib" ] \
      && command -v uv >/dev/null 2>&1; then
    if ! PYTHON=$(uv run --locked --project "$ROOT" python -c 'import sys; print(sys.executable)') \
        || [ -z "$PYTHON" ]; then
      echo "$_ora_python_name: uv run --locked could not prepare the checkout's environment ($ROOT/.venv); see uv's message above" >&2
      exit 1
    fi
  else
    PYTHON=python3
    if [ -z "$_ora_python_stdlib" ]; then
      if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
        _ora_python_why="\$UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT has no bin/python"
      elif [ -n "${UV_NO_SYNC:-}" ] && command -v uv >/dev/null 2>&1; then
        _ora_python_why="no $ROOT/.venv, \$PYTHON or \$UV_PROJECT_ENVIRONMENT, and \$UV_NO_SYNC keeps uv from creating one"
      else
        _ora_python_why="no $ROOT/.venv, \$PYTHON, \$UV_PROJECT_ENVIRONMENT or uv"
      fi
      echo "$_ora_python_name: no locked project environment ($_ora_python_why); using $(command -v python3 || echo python3) from PATH (uv sync --locked creates .venv)" >&2
    fi
  fi
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
}
