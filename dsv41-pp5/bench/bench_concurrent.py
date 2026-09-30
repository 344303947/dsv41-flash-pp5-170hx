#!/usr/bin/env python3
"""并发吞吐压测：N 个并发流式请求，测总吞吐/平均 TTFT/平均 decode。
用法: bench_concurrent.py N ctx_words gen
"""
import json
import random
import sys
import threading
import time
import urllib.request

KEY = "sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 500
gen = int(sys.argv[3]) if len(sys.argv) > 3 else 128
URL = "http://127.0.0.1:9004/v1/chat/completions"

res = [None] * N
lock = threading.Lock()


def one(i):
    nonce = f"X{i}-{random.randint(0, 10**12)}{time.time_ns()}"
    words = " ".join(f"w{random.randint(0, 999999)}" for _ in range(ctx))
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": f"[{nonce}] {words}\nWrite an essay."}],
        "max_tokens": gen, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time(); t_first = t_last = None; usage = None; ntok = 0
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
                if o.get("usage"):
                    usage = o["usage"]
                for ch in o.get("choices", []):
                    dd = ch.get("delta", {}) or {}
                    if dd.get("content") or dd.get("reasoning") or dd.get("reasoning_content"):
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now; ntok += 1
    except Exception as e:
        res[i] = {"err": str(e)[:120]}; return
    ctok = usage.get("completion_tokens") if usage else ntok
    res[i] = {
        "ttft": (t_first - t0) if t_first else None,
        "decode": ((ctok - 1) / (t_last - t_first)) if (ctok and t_last > t_first) else None,
        "ctok": ctok, "end": t_last or time.time(),
    }


t_start = time.time()
ths = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t_start

ok = [r for r in res if r and "err" not in r]
errs = [r for r in res if r and "err" in r]
tot_tok = sum(r["ctok"] for r in ok)
ttfts = [r["ttft"] for r in ok if r["ttft"]]
dec = [r["decode"] for r in ok if r["decode"]]
print(f"N={N} ctx={ctx} gen={gen}  wall={wall:.2f}s")
print(f"  完成={len(ok)} 失败={len(errs)}  总生成token={tot_tok}")
print(f"  聚合吞吐={tot_tok/wall:.1f} tok/s  平均TTFT={sum(ttfts)/len(ttfts)*1000:.0f}ms"
      f"  平均decode={sum(dec)/len(dec):.1f} tok/s")
if errs:
    print("  errs:", errs[:3])
