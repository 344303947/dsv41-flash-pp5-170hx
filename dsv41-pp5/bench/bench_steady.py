#!/usr/bin/env python3
"""稳态并发 decode 压测：所有请求都进入 decode 后，只在重叠窗口内统计，
排除 prefill 与排队干扰。区分「batch decode 退化」与「prefill 抢占」。

用法: bench_steady.py N ctx_words gen
"""
import json
import random
import sys
import threading
import time
import urllib.request

KEY = "sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 200
gen = int(sys.argv[3]) if len(sys.argv) > 3 else 512
URL = "http://127.0.0.1:9004/v1/chat/completions"
res = [None] * N


def one(i):
    nonce = f"X{i}-{random.randint(0, 10**12)}{time.time_ns()}"
    words = " ".join(f"w{random.randint(0, 999999)}" for _ in range(ctx))
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": f"[{nonce}] {words}\nWrite a long essay."}],
        "max_tokens": gen, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    ts = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:
                    continue
                for ch in o.get("choices", []):
                    dd = ch.get("delta", {}) or {}
                    if dd.get("content") or dd.get("reasoning") or dd.get("reasoning_content"):
                        ts.append(time.time())
    except Exception as e:
        res[i] = {"err": str(e)[:120]}; return
    res[i] = {"ts": ts, "t0": t0}


t_start = time.time()
ths = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t_start

ok = [r for r in res if r and "ts" in r and len(r["ts"]) >= 2]
if not ok:
    print(f"N={N}: no data", [r for r in res if r])
    sys.exit()

firsts = [r["ts"][0] for r in ok]
lasts = [r["ts"][-1] for r in ok]
ttft = [r["ts"][0] - r["t0"] for r in ok]
start = max(firsts)
end = min(lasts)
if end <= start:
    start = sum(firsts) / len(firsts)
    end = sum(lasts) / len(lasts)
win = end - start
in_win = sum(sum(1 for t in r["ts"] if start <= t <= end) for r in ok)
agg = in_win / win if win > 0 else float("nan")
print(f"N={N:2d} ctx={ctx} gen={gen}  wall={wall:.2f}s  稳态窗口={win:.2f}s")
print(f"   稳态聚合decode={agg:.1f} tok/s  每流={agg/N:.1f} tok/s"
      f"  平均TTFT={sum(ttft)/len(ttft)*1000:.0f}ms  总token={sum(len(r['ts']) for r in ok)}")
