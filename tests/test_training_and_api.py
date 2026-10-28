import json
from pathlib import Path
from denial_appeal.train_baseline import train_and_evaluate

def test_tiny_training_run(tiny_train, tmp_path):
    out = tmp_path / "artifacts"
    results = train_and_evaluate(str(tiny_train / "train.jsonl"),
                                 str(tiny_train / "test.jsonl"), str(out), seed=11)
    assert results["trained_model"]["n"] == 40
    assert results["trained_model"]["accuracy"] >= 0.9  # templated synthetic
    assert (out / "baseline_model.pkl").exists()
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["seed"] == 11 and len(manifest["train_data_sha256"]) == 64

def test_trained_beats_or_matches_rules_gate(tiny_train, tmp_path):
    results = train_and_evaluate(str(tiny_train / "train.jsonl"),
                                 str(tiny_train / "test.jsonl"),
                                 str(tmp_path / "a"), seed=11)
    assert results["trained_model"]["accuracy"] >= results["rules_baseline"]["accuracy"]

def _client(monkeypatch, tiny_train, tmp_path):
    out = tmp_path / "a"
    train_and_evaluate(str(tiny_train / "train.jsonl"),
                       str(tiny_train / "test.jsonl"), str(out), seed=11)
    monkeypatch.setenv("BASELINE_MODEL", str(out / "baseline_model.pkl"))
    import importlib
    import denial_appeal.serve_stub as srv
    importlib.reload(srv)
    from fastapi.testclient import TestClient
    return TestClient(srv.app), srv

def test_api_healthz(monkeypatch, tiny_train, tmp_path):
    client, _ = _client(monkeypatch, tiny_train, tmp_path)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_api_models(monkeypatch, tiny_train, tmp_path):
    client, _ = _client(monkeypatch, tiny_train, tmp_path)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "denial-appeal-finetune-baseline" in ids

def test_api_classify(monkeypatch, tiny_train, tmp_path, records):
    client, _ = _client(monkeypatch, tiny_train, tmp_path)
    r = client.post("/classify", json={"denial_letter": records[0]["denial_letter"]})
    assert r.status_code == 200 and r.json()["cause"] == records[0]["cause"]

def test_api_chat_completions_shape(monkeypatch, tiny_train, tmp_path, records):
    client, _ = _client(monkeypatch, tiny_train, tmp_path)
    r = client.post("/v1/chat/completions", json={
        "model": "denial-appeal-finetune-baseline",
        "messages": [{"role": "user", "content": records[0]["denial_letter"]}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    payload = json.loads(body["choices"][0]["message"]["content"])
    assert payload["cause"] == records[0]["cause"]
    assert "appeal_letter" in payload

def test_lora_config_and_dry_run_plan():
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(root / "train_lora.py"),
                           "--config", str(root / "configs/lora_qwen25_05b.yaml"),
                           "--dry-run"], capture_output=True, text=True, cwd=root)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan["train_examples"] == 600
