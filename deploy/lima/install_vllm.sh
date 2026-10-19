#!/usr/bin/env bash
# Install vLLM (CPU wheel, x86 AVX2/AVX512) inside the nafs-vllm Lima VM. Run inside the VM:
#   limactl shell --workdir /nafs-aql nafs-vllm -- bash deploy/lima/install_vllm.sh
set -euxo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.31.0}
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libtcmalloc-minimal4 libnuma1 python3-venv python3-dev build-essential curl ca-certificates  # g++ is required by torch.compile (inductor) on the CPU backend
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --python 3.12 "$HOME/vllm-env"
source "$HOME/vllm-env/bin/activate"
uv pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_39_x86_64.whl" --torch-backend cpu
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'avx512_bf16', torch.cpu._is_avx512_bf16_supported())"
find "$HOME/vllm-env" -name "libiomp5.so" | head -1 > "$HOME/.iomp_path"
echo "IOMP: $(cat $HOME/.iomp_path)"
