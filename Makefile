PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: data train eval test serve bench lora-dry-run lora-smoke
data:
	$(PY) scripts/generate_data.py
train:
	$(PY) scripts/train_baseline_cli.py
eval:
	$(PY) scripts/evaluate_cli.py
test:
	$(PY) -m pytest -q
serve:
	.venv/bin/uvicorn denial_appeal.serve_stub:app --port 8000
bench:
	$(PY) bench/benchmark.py --requests 50
lora-dry-run:
	$(PY) train_lora.py --config configs/lora_qwen25_05b.yaml --dry-run
lora-smoke:
	$(PY) train_lora.py --config configs/lora_qwen25_05b.yaml --smoke
