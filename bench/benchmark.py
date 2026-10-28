#!/usr/bin/env python3
"""Latency/throughput benchmark against any running compatible server
(vLLM or the CPU fallback). Uses only the standard library."""
from __future__ import annotations

import argparse, json, statistics, time, urllib.request

PROMPT = ("Analyse this denial letter. Respond with JSON containing cause, "
          "appeal_success_probability_band, and appeal_letter.\n\n"
          "Claim SYN-1 for outpatient physical therapy visit has been denied "
          "(code DEN-197). Reason: the service is precertification absent. "
          "No authorization on file was found. SYNTHETIC DATA.")


def one_request(base: str, model: str) -> float:
    body = json.dumps({"model": model, "messages": [
        {"role": "user", "content": PROMPT}], "temperature": 0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        json.loads(resp.read())
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="denial-appeal-finetune-baseline")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    for _ in range(a.warmup):
        one_request(a.base_url, a.model)
    lat = []
    t0 = time.perf_counter()
    for _ in range(a.requests):
        lat.append(one_request(a.base_url, a.model))
    wall = time.perf_counter() - t0
    lat_ms = sorted(x * 1000 for x in lat)
    def pct(p): return round(lat_ms[min(len(lat_ms) - 1, int(p * len(lat_ms)))], 2)
    result = {"server": a.base_url, "model": a.model, "requests": a.requests,
              "p50_ms": pct(0.50), "p95_ms": pct(0.95),
              "mean_ms": round(statistics.mean(lat_ms), 2),
              "throughput_req_per_sec": round(a.requests / wall, 2)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
