#!/usr/bin/env python3
"""Debug what chunks the client.generate() yields in stream mode."""
import asyncio, time, sys
sys.path.insert(0, "/home/ubuntu/sglang-omni")

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.config import compile_pipeline
from sglang_omni.config.manager import ConfigManager

async def test():
    # We need to connect to the running server's coordinator
    # Instead, let's just test via HTTP and add debug logging
    import httpx
    payload = {
        "input": "Test.",
        "voice": "default",
        "response_format": "wav",
        "ref_audio": "/home/ubuntu/perla1.wav",
        "ref_text": "",
    }
    
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        async with client.stream("POST", "http://127.0.0.1:8000/v1/audio/speech/stream", json=payload, timeout=60) as resp:
            i = 0
            async for chunk in resp.aiter_bytes():
                t = (time.perf_counter() - t0) * 1000
                print(f"  chunk {i}: t={t:.0f}ms size={len(chunk)}B")
                i += 1
        print(f"  Total: {(time.perf_counter()-t0)*1000:.0f}ms, chunks={i}")

asyncio.run(test())
