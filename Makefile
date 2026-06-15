# CephSight — Makefile for VM operations
# Usage: make setup | make serve | make analyze URL=<pulpito-url>

MODEL_NAME    = cephsight
MODEL_BASE    = qwen3:14b
MODEL_CTX     = 32768
BACKEND_DIR   = backend
FRONTEND_DIR  = frontend
OUTPUT_DIR    = output

.PHONY: setup pull create-model serve analyze clean-cache help

help:
	@echo "CephSight — AI-Powered Teuthology CI Failure Analyzer"
	@echo ""
	@echo "  make setup         Pull base model + create custom model (run once)"
	@echo "  make serve         Start the Flask API server"
	@echo "  make analyze URL=  Run CLI analysis on a Pulpito URL"
	@echo "  make clean-cache   Clear analysis cache"
	@echo ""
	@echo "Model: $(MODEL_BASE) | Context: $(MODEL_CTX) | Temperature: 0.1"

setup: pull create-model
	@echo "Setup complete. Run 'make serve' to start."

pull:
	@echo "Pulling base model $(MODEL_BASE)..."
	ollama pull $(MODEL_BASE)

create-model:
	@echo "Creating custom model '$(MODEL_NAME)' with $(MODEL_CTX) context..."
	cd $(BACKEND_DIR) && ollama create $(MODEL_NAME) -f Modelfile
	@echo "Model '$(MODEL_NAME)' created. Verify with: ollama list"

serve:
	@echo "Starting CephSight backend..."
	cd $(BACKEND_DIR) && python3 app.py

analyze:
ifndef URL
	$(error URL is required. Usage: make analyze URL="https://pulpito.ceph.com/...")
endif
	cd $(BACKEND_DIR) && python3 main.py "$(URL)" --model $(MODEL_NAME) --clear-cache -o $(OUTPUT_DIR)

clean-cache:
	rm -rf $(BACKEND_DIR)/output/.analysis_cache
	@echo "Cache cleared."
