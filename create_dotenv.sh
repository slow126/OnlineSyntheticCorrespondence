#!/usr/bin/env bash
set -euo pipefail

# create_dotenv.sh
# Usage examples:
#   # Use active env, auto-detect CUDA:
#   bash create_dotenv.sh
#
#   # Specify env name:
#   bash create_dotenv.sh --env myenv
#
#   # Force system CUDA:
#   bash create_dotenv.sh --mode system
#
#   # Force conda CUDA for a specific env, write to custom path:
#   bash create_dotenv.sh --env myenv --mode conda --output /path/to/.env
#
#   # Preview without writing:
#   bash create_dotenv.sh --dry-run

ENV_NAME=""
MODE="auto"          # auto | conda | system
OUT_FILE=".env"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="${2:-}"; shift 2;;
    --mode) MODE="${2:-}"; shift 2;;
    --output) OUT_FILE="${2:-}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help)
      sed -n '1,100p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# --- Resolve conda env prefix ---
resolve_env_prefix() {
  local name="$1"
  if [[ -n "$name" ]]; then
    # Try conda to resolve env path robustly
    if command -v conda >/dev/null 2>&1; then
      # Try "conda env list" first
      local p
      p="$(conda env list | awk -v n="^${name}\$" 'NF>=2 { if ($1 ~ n) print $NF }' | head -n1)"
      if [[ -z "$p" ]]; then
        # Fallback: conda run to echo CONDA_PREFIX
        p="$(conda run -n "$name" python - <<'PY'
import os, sys
print(os.environ.get("CONDA_PREFIX",""))
PY
)"
      fi
      [[ -n "$p" ]] && echo "$p" && return 0
    fi
    echo "Error: could not resolve conda env path for '$name'." >&2
    return 1
  else
    # Use active env
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      echo "$CONDA_PREFIX"
      return 0
    fi
    echo "Error: no env name given and no active conda env (CONDA_PREFIX empty)." >&2
    return 1
  fi
}

ENV_PREFIX="$(resolve_env_prefix "$ENV_NAME")"

# --- Detect CUDA root based on mode ---
detect_cuda_root() {
  local mode="$1"
  local env_prefix="$2"

  if [[ "$mode" == "conda" ]]; then
    echo "$env_prefix"; return 0
  fi

  if [[ "$mode" == "system" ]]; then
    if [[ -d /usr/local/cuda ]]; then
      echo "/usr/local/cuda"; return 0
    fi
    # Pick highest /usr/local/cuda-*
    local best
    best="$(ls -1d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -n "$best" ]]; then echo "$best"; return 0; fi
    echo ""; return 0
  fi

  # mode=auto: prefer conda CUDA if looks present
  if [[ -d "$env_prefix/bin" ]] && (command -v "$env_prefix/bin/nvcc" >/dev/null 2>&1 || [[ -d "$env_prefix/lib" ]] || [[ -d "$env_prefix/lib64" ]]); then
    echo "$env_prefix"; return 0
  fi
  if [[ -d /usr/local/cuda ]]; then
    echo "/usr/local/cuda"; return 0
  fi
  local best
  best="$(ls -1d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -n "$best" ]]; then echo "$best"; return 0; fi

  echo ""; return 0
}

CUDA_ROOT="$(detect_cuda_root "$MODE" "$ENV_PREFIX")"

# --- Compose .env contents (Linux). macOS CUDA is deprecated; Windows uses different separators. ---
emit_env() {
  local cuda_root="$1"
  local osname; osname="$(uname)"
  if [[ -z "$cuda_root" ]]; then
    cat <<'EOF'
# No CUDA installation detected. Remove lines you don't need or set manually.
# CUDA_HOME=
# CUDA_PATH=
# PATH=${PATH}
# LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
EOF
    return 0
  fi

  if [[ "$osname" == "Linux" ]]; then
    local libdir=""
    if [[ -d "$cuda_root/lib64" ]]; then libdir="$cuda_root/lib64"; elif [[ -d "$cuda_root/lib" ]]; then libdir="$cuda_root/lib"; fi
    cat <<EOF
CUDA_HOME=$cuda_root
CUDA_PATH=$cuda_root
PATH=$cuda_root/bin:\${PATH}
${libdir:+LD_LIBRARY_PATH=$libdir:\${LD_LIBRARY_PATH}}
EOF
  elif [[ "$osname" == "Darwin" ]]; then
    # CUDA on macOS is effectively not supported (post-10.13), but we still emit PATH vars if someone has a legacy install.
    cat <<EOF
CUDA_HOME=$cuda_root
CUDA_PATH=$cuda_root
PATH=$cuda_root/bin:\${PATH}
# Note: macOS typically doesn't use LD_LIBRARY_PATH for CUDA; frameworks are different.
EOF
  else
    # Windows (Git Bash) users should adapt manually; conda/VS Code handle env differently on Windows.
    cat <<EOF
# Detected OS: $osname
# Adapt for Windows manually if needed:
# CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x
# PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin;%PATH%
EOF
  fi
}

CONTENT="$(emit_env "$CUDA_ROOT")"

echo "Resolved conda env: $ENV_PREFIX"
[[ -n "$CUDA_ROOT" ]] && echo "Using CUDA root: $CUDA_ROOT" || echo "CUDA root not found; writing placeholder .env"

echo
echo "---- .env preview ($OUT_FILE) ----"
echo "$CONTENT"
echo "----------------------------------"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(dry-run) Not writing file."
  exit 0
fi

printf "%s\n" "$CONTENT" > "$OUT_FILE"
echo "Wrote $OUT_FILE"

# Optional: suggest VS Code setting if not present
if [[ ! -f ".vscode/settings.json" ]]; then
  echo
  echo "Tip: create .vscode/settings.json with:"
  cat <<EOF
{
  "python.envFile": "\${workspaceFolder}/$OUT_FILE"
}
EOF
fi
