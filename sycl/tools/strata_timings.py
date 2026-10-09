#!/usr/bin/env python3
"""strata_timings.py: per-prompt speed of a served Strata engine, from the API's own clock.

Posts each prompt of a JSON suite to <base>/v1/chat/completions (not streamed, temperature 0, thinking off) and
prints one line per prompt: id, prompt and completion tokens, every numeric field of the response's `timings`
object (serve/server.py's, in llama.cpp's names: prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms,
predicted_per_second, cache_n, draft_n, draft_n_accepted, ...; whatever the server sends is printed) and the
wall-clock seconds, then a median row. Written for the 12-prompt suite in the operator's homelab repo
(tools/b70-tuning/prompts/suite.json): {"version": 1, "sampling": {...}, "prompts": [{"id", "category",
"max_tokens", "messages": [...], "tools"?: [...]}]}. A plain list of {"id", "prompt"} objects works too, and a
prompt's own "max_tokens" wins over --max-tokens. Standard library only.

    python3 sycl/tools/strata_timings.py --suite suite.json [--base http://127.0.0.1:8097] [--repeat 3] [--out raw.jsonl]

STRATA_API_KEY in the environment becomes the Authorization bearer when set.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


def load_suite(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("prompts", []) if isinstance(data, dict) else data
    out = []
    for i, p in enumerate(items):
        if not isinstance(p, dict) or not (p.get("messages") or p.get("prompt")):
            raise SystemExit(f"{path}: prompt {i} has neither \"messages\" nor \"prompt\"")
        out.append({"id": str(p.get("id", i)), "max_tokens": p.get("max_tokens"), "tools": p.get("tools"),
                    "messages": p.get("messages") or [{"role": "user", "content": p["prompt"]}]})
    return out


def ask(base: str, p: dict, max_tokens: int, key: str) -> tuple[dict, float]:
    body = {"messages": p["messages"], "stream": False, "temperature": 0,
            "max_tokens": int(p["max_tokens"] or max_tokens), "chat_template_kwargs": {"enable_thinking": False}}
    if p.get("tools"):
        body["tools"] = p["tools"]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{p['id']}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
    except (OSError, ValueError) as e:
        raise SystemExit(f"{p['id']}: {e}")
    return res, time.monotonic() - t0


def numeric(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def fmt(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8097", help="the server (default: %(default)s)")
    ap.add_argument("--suite", required=True, help="the JSON prompt suite")
    ap.add_argument("--max-tokens", type=int, default=256, help="when a prompt names none (default: %(default)s)")
    ap.add_argument("--repeat", type=int, default=1, metavar="N", help="run the suite N times, repeats interleaved")
    ap.add_argument("--out", help="append the raw responses (message content removed) here, one JSON per line")
    a = ap.parse_args()
    key = os.environ.get("STRATA_API_KEY", "")
    prompts = load_suite(a.suite)
    rows, keys = [], []
    out = open(a.out, "a", encoding="utf-8") if a.out else None
    for n in range(max(1, a.repeat)):
        for p in prompts:
            res, wall = ask(a.base, p, a.max_tokens, key)
            usage, timings = res.get("usage") or {}, numeric(res.get("timings"))
            for k in timings:
                if k not in keys:
                    keys.append(k)
            row = {"id": p["id"], "run": n + 1, "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens"), **timings, "wall_s": round(wall, 2)}
            rows.append(row)
            if not rows[:-1]:
                print("\t".join(["id", "run", "prompt_tokens", "completion_tokens", *keys, "wall_s"]))
            print("\t".join(fmt(row.get(k)) for k in ["id", "run", "prompt_tokens", "completion_tokens", *keys, "wall_s"]),
                  flush=True)
            if out:
                for c in res.get("choices") or []:
                    c.pop("message", None)
                out.write(json.dumps({"id": p["id"], "run": n + 1, "wall_s": round(wall, 2), "response": res}) + "\n")
                out.flush()
    if out:
        out.close()
    cols = ["prompt_tokens", "completion_tokens", *keys, "wall_s"]
    med = {k: statistics.median(vs) for k in cols if (vs := [r[k] for r in rows if isinstance(r.get(k), (int, float))])}
    print("\t".join(["median", str(len(rows)), *(fmt(med.get(k)) for k in cols)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
