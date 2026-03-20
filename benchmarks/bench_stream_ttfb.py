#!/usr/bin/env python3
"""Streaming TTS benchmark with TTFB measurement."""
import asyncio, time, statistics
import httpx

URL = "http://127.0.0.1:8000/v1/audio/speech/stream"
REF_AUDIO = "/home/ubuntu/perla1.wav"
TEXT = "Bonjour, je suis Perla, une assistante vocale intelligente. Comment puis-je vous aider aujourd'hui? Je peux répondre à vos questions sur de nombreux sujets différents."

async def single_request(client, req_id):
    payload = {
        "input": TEXT,
        "voice": "default",
        "response_format": "wav",
        "ref_audio": REF_AUDIO,
        "ref_text": "",
    }
    t0 = time.perf_counter()
    ttfb = None
    total_bytes = 0
    chunks = 0
    try:
        async with client.stream("POST", URL, json=payload, timeout=120) as resp:
            async for chunk in resp.aiter_bytes():
                if ttfb is None:
                    ttfb = (time.perf_counter() - t0) * 1000
                total_bytes += len(chunk)
                chunks += 1
    except Exception as e:
        total_t = (time.perf_counter() - t0) * 1000
        return {"id": req_id, "error": str(e)[:100], "total_ms": total_t, "ttfb_ms": ttfb, "chunks": chunks, "total_bytes": total_bytes}
    
    total_t = (time.perf_counter() - t0) * 1000
    return {
        "id": req_id,
        "ttfb_ms": ttfb,
        "total_ms": total_t,
        "chunks": chunks,
        "total_bytes": total_bytes,
        "error": None,
    }

async def run_batch(batch_size):
    print(f"\n{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    async with httpx.AsyncClient() as client:
        if batch_size == 1:
            print("Warmup...")
            await single_request(client, "warmup")
        
        tasks = [single_request(client, i) for i in range(batch_size)]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        batch_time = (time.perf_counter() - t0) * 1000
    
    ttfbs = []
    totals = []
    errors = 0
    for r in results:
        ttfb_str = f"{r['ttfb_ms']:.0f}ms" if r['ttfb_ms'] is not None else "N/A"
        if r["error"]:
            errors += 1
            print(f"  Request {r['id']}: TTFB={ttfb_str} Total={r['total_ms']:.0f}ms ERROR: {r['error']}")
        else:
            if r['ttfb_ms'] is not None:
                ttfbs.append(r['ttfb_ms'])
            totals.append(r['total_ms'])
            print(f"  Request {r['id']}: TTFB={ttfb_str} Total={r['total_ms']:.0f}ms Chunks={r['chunks']} Bytes={r['total_bytes']}")
    
    print(f"\n  Summary:")
    print(f"    Successful: {len(totals)}/{batch_size}, Errors: {errors}")
    if ttfbs:
        print(f"    TTFB  - min={min(ttfbs):.0f}ms avg={statistics.mean(ttfbs):.0f}ms max={max(ttfbs):.0f}ms")
    if totals:
        print(f"    Total - min={min(totals):.0f}ms avg={statistics.mean(totals):.0f}ms max={max(totals):.0f}ms")
    print(f"    Batch wall time: {batch_time:.0f}ms")

async def main():
    for batch_size in [1, 5, 8]:
        await run_batch(batch_size)

asyncio.run(main())
