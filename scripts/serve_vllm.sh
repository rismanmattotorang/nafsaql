#!/usr/bin/env bash
# Host wrapper: run vLLM (CPU backend) in the nafs-vllm Lima VM. No Docker involved.
#   scripts/serve_vllm.sh setup     create+start the VM and install vLLM (once)
#   scripts/serve_vllm.sh start     start vLLM on :8101 (Qwen3-1.7B BF16 serving System 1 and System 2 modes; fits a 32 GB host)
#   scripts/serve_vllm.sh start-s1|start-s2   two-model layout for hosts with >= 48 GB RAM
#   scripts/serve_vllm.sh stop|status|vm-stop
set -euo pipefail
cd "$(dirname "$0")/.."
VM=nafs-vllm
case "${1:-status}" in
  setup) limactl list -q | grep -qx "$VM" || limactl create --name="$VM" --tty=false deploy/lima/nafs-vllm.yaml
         limactl start "$VM" --tty=false 2>/dev/null || true
         limactl shell --workdir /nafs-aql "$VM" -- bash deploy/lima/install_vllm.sh ;;
  vm-stop) limactl stop "$VM" ;;
  start|start-single|start-s1|start-s2|stop|status)
         limactl list "$VM" --format '{{.Status}}' | grep -q Running || limactl start "$VM" --tty=false
         limactl shell --workdir /nafs-aql "$VM" -- bash deploy/lima/serve_vllm.sh "$1"
         if [[ "$1" == start* ]]; then for p in $( [ "$1" = start-s2 ] && echo 8102 || echo 8101); do
           printf "waiting for :%s " $p; for i in $(seq 1 900); do curl -sf "localhost:$p/v1/models" >/dev/null && { echo ok; break; }; sleep 2; printf .; done
           name=$( [ "$1" = start-s1 ] && echo system1 || ([ $p = 8102 ] && echo system2) || echo qwen3-1.7b )
           printf "warming up %s (first-use compile, can take several minutes on CPU) " $name
           curl -sf -m 1800 "localhost:$p/v1/chat/completions" -H 'Content-Type: application/json' \
             -d "{\"model\":\"$name\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"logprobs\":true,\"top_logprobs\":1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" >/dev/null && echo done || echo "warm-up failed (see logs/vllm_$name.log)"
         done; fi ;;
  *) echo "usage: $0 setup|start|start-s1|start-s2|stop|status|vm-stop"; exit 2 ;;
esac
