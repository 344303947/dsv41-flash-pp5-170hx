#!/usr/bin/env python3
"""DeepSeek-V4.1 冷 prefill / decode 测量（streaming，随机 nonce 防前缀命中）。

用法: bench_v41.py [ctx_words] [gen_tokens] [port]
  ctx_words : 伪随机词数（约等于 token 数）
  gen_tokens: 生成 token 上限
"""
import json
import random
import sys
import time
import urllib.request

KEY = "sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807"
ctx_words = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
gen = int(sys.argv[2]) if len(sys.argv) > 2 else 128
port = sys.argv[3] if len(sys.argv) > 3 else "9004"
URL = f"http://127.0.0.1:{port}/v1/chat/completions"


def run():
    nonce = f"X{random.randint(0, 10**12)}{time.time_ns()}"
    words = " ".join(f"w{random.randint(0, 999999)}" for _ in range(ctx_words))
    prompt = (
        f"[{nonce}] {words}\n\n"
        "Write a long, detailed technical essay about computing systems."
    )
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
    )
    t0 = time.time()
    t_first = t_last = None
    usage = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}) or {}
                if (
                    d.get("content")
                    or d.get("reasoning")
                    or d.get("reasoning_content")
                ):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    ntok += 1
    if t_first is None:
        print("ERROR: no tokens received (prompt may have failed)")
        return None
    ptok = usage.get("prompt_tokens") if usage else None
    ctok = usage.get("completion_tokens") if usage else ntok
    ttft = t_first - t0
    dec_dt = t_last - t_first
    prefill = (ptok / ttft) if ptok else float("nan")
    decode = ((ctok - 1) / dec_dt) if (ctok and dec_dt > 0) else float("nan")
    print(
        f"ctx_words={ctx_words} gen={gen}\n"
        f"  prompt_tokens={ptok} completion_tokens={ctok}\n"
        f"  TTFT={ttft*1000:.0f} ms  cold_prefill={prefill:.0f} tok/s\n"
        f"  decode={decode:.2f} tok/s  (decode_window={dec_dt:.2f}s)"
    )
    return {"prompt_tokens": ptok, "completion_tokens": ctok,
            "ttft_ms": ttft * 1000, "prefill": prefill, "decode": decode}


if __name__ == "__main__":
    run()
