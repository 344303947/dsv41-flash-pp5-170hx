#!/usr/bin/env python3
"""混合压测：1 个长 prefill 请求 + N 个短请求，测短请求被长 prefill 拖累的程度。
用法: bench_mixed.py N_short long_ctx_words short_ctx_words
"""
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

KEY = "sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
long_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
short_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 50
URL = "http://127.0.0.1:9004/v1/chat/completions"
res = {}


def call(tag, ctx, gen, t0):
    nonce = f"{tag}-{random.randint(0, 10**12)}{time.time_ns()}"
    words = " ".join(f"w{random.randint(0, 999999)}" for _ in range(ctx))
    body = {"model": "DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": f"[{nonce}] {words}\nWrite."}],
            "max_tokens": gen, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t_first = None
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
                    if (dd.get("content") or dd.get("reasoning")
                            or dd.get("reasoning_content")) and t_first is None:
                        t_first = time.time() - t0
    except Exception as e:
        res[tag] = {"err": str(e)[:100]}; return
    res[tag] = {"ttft": t_first}


t0 = time.time()
# 长 prefill 先发
threading.Thread(target=call, args=("long", long_ctx, 16, t0), daemon=True).start()
time.sleep(0.3)  # 让长请求先进调度器
# 短请求
ths = [threading.Thread(target=call, args=(f"s{i}", short_ctx, 64, t0)) for i in range(N)]
for t in ths:
    t.start()
for t in ths:
    t.join()
long_th = threading.Thread(target=lambda: None)  # 长请求后台继续

shorts = [res[f"s{i}"]["ttft"] for i in range(N) if res.get(f"s{i}") and "ttft" in res[f"s{i}"]]
print(f"N_short={N} long_ctx={long_ctx} short_ctx={short_ctx}")
print(f"  短请求 TTFT: mean={statistics.mean(shorts)*1000:.0f}ms  "
      f"min={min(shorts)*1000:.0f}  max={max(shorts)*1000:.0f}  n={len(shorts)}")
