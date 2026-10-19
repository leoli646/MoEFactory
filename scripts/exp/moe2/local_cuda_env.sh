#!/usr/bin/env bash

# Configure the CUDA toolkit for local/rlaunch shells. PyTorch wheels include
# the CUDA runtime, but DeepSpeed also probes nvcc while it is being imported.
setup_local_cuda_env() {
  local python_bin="${1:-python3}"
  local require_toolkit="${2:-1}"

  if [[ -d /usr/local/nvidia/bin ]]; then
    case ":${PATH:-}:" in
      *:/usr/local/nvidia/bin:*) ;;
      *) export PATH="/usr/local/nvidia/bin:${PATH:-}" ;;
    esac
  fi
  if [[ -d /usr/local/nvidia/lib64 ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *:/usr/local/nvidia/lib64:*) ;;
      *) export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi

  local torch_cuda
  if ! torch_cuda="$("$python_bin" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)"; then
    if [[ "$require_toolkit" == "1" ]]; then
      echo "Unable to query torch.version.cuda with $python_bin." >&2
      return 2
    fi
    return 0
  fi
  if [[ -z "$torch_cuda" ]]; then
    if [[ "$require_toolkit" == "1" ]]; then
      echo "The selected PyTorch build has no CUDA support: $python_bin" >&2
      return 2
    fi
    return 0
  fi

  local torch_cuda_major="${torch_cuda%%.*}"
  local requested_cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
  local nvcc_path=""
  local -a candidates=()
  [[ -n "$requested_cuda_home" ]] && candidates+=("$requested_cuda_home")

  nvcc_path="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$nvcc_path" ]]; then
    nvcc_path="$(readlink -f "$nvcc_path" 2>/dev/null || printf '%s' "$nvcc_path")"
    candidates+=("$(dirname "$(dirname "$nvcc_path")")")
  fi
  candidates+=(/usr/local/cuda)

  local candidate
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && candidates+=("$candidate")
  done < <(compgen -G '/usr/local/cuda-*' | sort -Vr)
  [[ -n "${CONDA_PREFIX:-}" ]] && candidates+=("$CONDA_PREFIX")
  [[ -n "${MOE_ENV:-}" ]] && candidates+=("$MOE_ENV")

  local selected_cuda_home=""
  local selected_nvcc_version=""
  local nvcc_output nvcc_major nvcc_minor
  for candidate in "${candidates[@]}"; do
    candidate="${candidate%/}"
    [[ -x "$candidate/bin/nvcc" ]] || continue
    nvcc_output="$("$candidate/bin/nvcc" -V 2>/dev/null || true)"
    if [[ "$nvcc_output" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
      nvcc_major="${BASH_REMATCH[1]}"
      nvcc_minor="${BASH_REMATCH[2]}"
    else
      continue
    fi
    if [[ "$nvcc_major" == "$torch_cuda_major" ]]; then
      selected_cuda_home="$candidate"
      selected_nvcc_version="$nvcc_major.$nvcc_minor"
      break
    fi
  done

  if [[ -z "$selected_cuda_home" ]]; then
    if [[ "$require_toolkit" == "1" ]]; then
      echo "No CUDA toolkit matching PyTorch CUDA $torch_cuda was found." >&2
      echo "Set CUDA_HOME to a toolkit containing bin/nvcc, or launch with the same CUDA image as the working RJob." >&2
      [[ -n "$requested_cuda_home" ]] && echo "Requested CUDA_HOME was: $requested_cuda_home" >&2
      return 2
    fi
    return 0
  fi

  if [[ -n "$requested_cuda_home" && "${requested_cuda_home%/}" != "$selected_cuda_home" ]]; then
    echo "WARNING: ignoring unusable CUDA_HOME=$requested_cuda_home; using $selected_cuda_home." >&2
  fi
  export CUDA_HOME="$selected_cuda_home"
  export CUDA_PATH="$selected_cuda_home"
  case ":${PATH:-}:" in
    *:"$CUDA_HOME/bin":*) ;;
    *) export PATH="$CUDA_HOME/bin:${PATH:-}" ;;
  esac
  echo "CUDA toolkit preflight: torch_cuda=$torch_cuda nvcc=$selected_nvcc_version CUDA_HOME=$CUDA_HOME"
}
