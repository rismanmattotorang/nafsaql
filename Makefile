# Nafs-Aql developer entry points.
TECTONIC := $(abspath tools/tectonic)

PHONY_EXTRA = acceptance tazkiya-build regression-suite
.PHONY: help up down up-vllm down-vllm build test $(PHONY_EXTRA) contract rooms-test docs report slides spec roadmap tables validate models models-vllm datasets vm-setup clean

help:
	@echo "Run the system (Phase 1):"
	@echo "  make models        download GGUF weights for the native llama.cpp backend (~3.6 GB)"
	@echo "  make up            build, start llama-server (S1 :8001, S2 :8002) and the gateway -> http://127.0.0.1:8080"
	@echo "  make down          stop gateway and llama-server"
	@echo "  make vm-setup      create the nafs-vllm Lima VM and install vLLM (CPU, no Docker)"
	@echo "  make models-vllm   download Hugging Face weights for vLLM"
	@echo "  make up-vllm       start vLLM :8101 in the VM (Qwen3-1.7B BF16, both modes), switch the gateway to it"
	@echo "Develop:"
	@echo "  make build | test | contract | datasets | validate"
	@echo "Documents:"
	@echo "  make docs | tables | clean"

build:
	cd rust && cargo build --release

test:
	cd rust && cargo test --release --workspace
	python3 -m unittest discover -s training -p "test_*.py"

acceptance:      ## MVP acceptance tests against the running stack (A9 restarts System 2)
	python3 -I scripts/mvp_acceptance.py

tazkiya-build:   ## Phase 3: build the distillation set from the ledger (see training/RUNBOOK.md)
	python3 -I training/tazkiya.py build --out data/tazkiya/

regression-suite: ## ER-008: the fixed 500-item regression suite
	python3 -I training/tazkiya.py regression-suite

contract:
	python3 -I scripts/contract_test.py --live http://127.0.0.1:8080

rooms-test:      ## live test of Rooms against the running stack (add SLOW=1 for the heartbeat, capacity and queue checks)
	python3 -I scripts/rooms_test.py $(if $(SLOW),--slow,)

up: build
	scripts/serve_llama.sh start
	scripts/gateway.sh start
	@echo "Open http://127.0.0.1:8080"

down:
	scripts/gateway.sh stop
	scripts/serve_llama.sh stop

vm-setup:
	scripts/serve_vllm.sh setup

up-vllm:
	scripts/serve_vllm.sh start
	curl -s -X POST localhost:8080/v1/system/backend -H 'Content-Type: application/json' -d '{"name":"vllm"}'; echo

down-vllm:
	scripts/serve_vllm.sh stop
	curl -s -X POST localhost:8080/v1/system/backend -H 'Content-Type: application/json' -d '{"name":"llama"}'; echo

models:
	scripts/fetch_models.sh profile-a
models-vllm:
	scripts/fetch_models.sh vllm
datasets:
	python3 -I scripts/build_datasets.py

docs: tables report slides spec roadmap
report:
	cd docs/report  && $(TECTONIC) nafs_aql_report.tex
slides:
	cd docs/slides  && $(TECTONIC) nafs_aql_slides.tex
spec: tables
	cd docs/spec    && $(TECTONIC) nafs_aql_spec.tex
roadmap: tables
	cd docs/roadmap && $(TECTONIC) nafs_aql_roadmap.tex
tables:
	python3 scripts/gen_tables.py

validate:
	scripts/validate_artifacts.sh

clean:
	find docs -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.xdv' -o -name '*.out' | xargs rm -f
	find . -name .DS_Store -not -path './third_party/*' -delete
	rm -rf scripts/__pycache__
