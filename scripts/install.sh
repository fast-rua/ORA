#!/bin/sh
# Install ORA for one user, without sudo:
#
#   curl -fsSL https://raw.githubusercontent.com/fast-rua/ORA/main/scripts/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- --ref v0.2.0      # a release tag, branch or commit
#
# It installs uv and Node.js 22 when they are missing, checks out ORA (a Git checkout, so every run
# can bind the exact commit it ran), syncs the locked Python environment and the agent core's
# packages, and puts an `ora` command in ~/.local/bin. Rerun it to update. Simulators are installed
# afterwards with `ora setup --env <env>`. Nothing outside the directories below is changed.
#
#   --ref REF        tag, branch or commit to check out (default: main, or ORA_REF)
#   --dir DIR        where ORA lives (default: ~/.local/share/ora, or ORA_HOME); the checkout is DIR/ORA
#   --repo URL       Git repository or bundle to install from (default: the public repository, or ORA_REPO)
#   --bin-dir DIR    where the `ora` command goes (default: ~/.local/bin, or ORA_BIN_DIR)
#   --no-doctor      skip the closing `ora doctor`
set -eu

NODE_VERSION=22.20.0      # the agent core needs Node.js 22.19 or newer
NODE_MIN=22.19
REPO=${ORA_REPO:-https://github.com/fast-rua/ORA.git}
REF=${ORA_REF:-main}
HOME_DIR=${ORA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/ora}
BIN_DIR=${ORA_BIN_DIR:-$HOME/.local/bin}
DOCTOR=1
MARK="# Installed by ORA's scripts/install.sh"

say() { printf 'ora-install: %s\n' "$*"; }
die() { printf 'ora-install: error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required: $2"; }

while [ $# -gt 0 ]; do
  case $1 in
    --ref) REF=${2:?--ref needs a value}; shift 2 ;;
    --dir) HOME_DIR=${2:?--dir needs a value}; shift 2 ;;
    --repo) REPO=${2:?--repo needs a value}; shift 2 ;;
    --bin-dir) BIN_DIR=${2:?--bin-dir needs a value}; shift 2 ;;
    --no-doctor) DOCTOR=0; shift ;;
    -h|--help) printf '%s\n' "usage: install.sh [--ref REF] [--dir DIR] [--repo URL] [--bin-dir DIR] [--no-doctor]"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done
CHECKOUT=$HOME_DIR/ORA

# --- platform and basic tools ---------------------------------------------------------------------
case $(uname -s) in
  Linux) OS=linux ;;
  Darwin) OS=darwin ;;
  *) die "ORA runs on Linux and macOS; this is $(uname -s)" ;;
esac
case $(uname -m) in
  x86_64|amd64) ARCH=x64 ;;
  aarch64|arm64) ARCH=arm64 ;;
  *) die "unsupported CPU architecture $(uname -m)" ;;
esac
need git "install it with your package manager (apt-get install git, brew install git)"
need curl "install it with your package manager"
need tar "install it with your package manager"
mkdir -p "$HOME_DIR" "$BIN_DIR"
ORIGINAL_PATH=$PATH
export PATH="$BIN_DIR:$PATH"

# --- uv ------------------------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  say "uv: $(command -v uv)"
else
  say "installing uv into $BIN_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BIN_DIR" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh >/dev/null
  command -v uv >/dev/null 2>&1 || die "uv did not install; see https://docs.astral.sh/uv/getting-started/installation/"
fi

# --- Node.js -------------------------------------------------------------------------------------
node_ok() {  # node_ok PATH: the binary runs and is at least NODE_MIN
  [ -x "$1" ] || return 1
  "$1" -e "const [a,b]=process.versions.node.split('.').map(Number),[x,y]='$NODE_MIN'.split('.').map(Number);process.exit(a>x||(a===x&&b>=y)?0:1)" 2>/dev/null
}
NODE=
for candidate in "${ORA_NODE:-}" "$(command -v node 2>/dev/null || true)" "$HOME/.local/bin/node" "$HOME_DIR/node/bin/node"; do
  if [ -n "$candidate" ] && node_ok "$candidate"; then NODE=$candidate; break; fi
done
if [ -n "$NODE" ]; then
  say "Node.js: $NODE ($("$NODE" --version))"
else
  need xz "unpacking Node.js needs xz (apt-get install xz-utils)"
  name=node-v$NODE_VERSION-$OS-$ARCH
  url=https://nodejs.org/dist/v$NODE_VERSION
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  say "installing Node.js $NODE_VERSION into $HOME_DIR/node"
  curl -fsSL "$url/$name.tar.xz" -o "$tmp/$name.tar.xz"
  curl -fsSL "$url/SHASUMS256.txt" -o "$tmp/SHASUMS256.txt"
  expected=$(grep " $name.tar.xz\$" "$tmp/SHASUMS256.txt" | cut -d' ' -f1)
  if command -v sha256sum >/dev/null 2>&1; then actual=$(sha256sum "$tmp/$name.tar.xz" | cut -d' ' -f1)
  else actual=$(shasum -a 256 "$tmp/$name.tar.xz" | cut -d' ' -f1); fi
  [ -n "$expected" ] && [ "$expected" = "$actual" ] || die "Node.js download does not match its published SHA-256"
  rm -rf "$HOME_DIR/node.new" && mkdir -p "$HOME_DIR/node.new"
  tar -xJf "$tmp/$name.tar.xz" -C "$HOME_DIR/node.new" --strip-components=1
  rm -rf "$HOME_DIR/node" && mv "$HOME_DIR/node.new" "$HOME_DIR/node"
  NODE=$HOME_DIR/node/bin/node
  node_ok "$NODE" || die "the installed Node.js does not run on this machine"
fi
NODE_BIN=$(dirname "$NODE")

# --- the checkout --------------------------------------------------------------------------------
if [ -d "$CHECKOUT/.git" ]; then
  [ -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=normal)" ] \
    || die "$CHECKOUT has local changes; commit or remove them, or install elsewhere with --dir"
  say "updating $CHECKOUT to $REF"
  git -C "$CHECKOUT" remote set-url origin "$REPO"
else
  [ ! -e "$CHECKOUT" ] || die "$CHECKOUT exists and is not a Git checkout"
  say "checking out ORA ($REF) into $CHECKOUT"
  git init --quiet "$CHECKOUT"
  git -C "$CHECKOUT" remote add origin "$REPO"
fi
git -C "$CHECKOUT" fetch --quiet --tags origin "$REF" || die "cannot fetch $REF from $REPO"
git -C "$CHECKOUT" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
COMMIT=$(git -C "$CHECKOUT" rev-parse --short=12 HEAD)

# --- Python environment and the agent core's packages --------------------------------------------
say "syncing the locked Python environment (uv installs Python 3.11 if needed)"
(cd "$CHECKOUT" && uv sync --all-extras --locked --quiet)
say "installing the agent core's packages"
PATH="$NODE_BIN:$PATH" npm_config_update_notifier=false npm ci --prefix "$CHECKOUT/ora-pi" --prefer-offline --no-audit --no-fund --loglevel=error >/dev/null

# --- the ora command -----------------------------------------------------------------------------
SHIM=$BIN_DIR/ora
if [ -e "$SHIM" ] && ! grep -q "$MARK" "$SHIM" 2>/dev/null; then
  die "$SHIM exists and was not written by this installer; move it, or pass --bin-dir"
fi
cat > "$SHIM" <<EOF
#!/bin/sh
$MARK; rerun it to update ORA.
export ORA_NODE="\${ORA_NODE:-$NODE}"
export ORA_COMMAND=ora
exec "$CHECKOUT/scripts/ora" "\$@"
EOF
chmod +x "$SHIM"
say "installed ORA $COMMIT: $SHIM -> $CHECKOUT"

case :$ORIGINAL_PATH: in
  *":$BIN_DIR:"*) ;;
  *) say "add $BIN_DIR to PATH for new shells: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc" ;;
esac

if [ "$DOCTOR" = 1 ]; then
  "$SHIM" doctor || say "ora doctor reported problems; they are listed above"
fi

cat <<EOF

Next:
  ora providers key DEEPSEEK_LOW_API_KEY     # once, asked without echo; ora providers lists the profiles
  ora run --env synthetic --provider-profile deepseek_flash_openai_low_16k_v1   # no simulator or GPU
  ora play --env synthetic                   # give a robot instructions yourself
  ora setup --env robodojo                   # install a simulator (libero, robotwin, robodojo)
  ora run --env robodojo --task general_pickup --confirm-launch
EOF
