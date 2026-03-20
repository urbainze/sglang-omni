# SGLang-Omni - Fish Audio S2-Pro (Fork optimise)

Fork de [sglang-omni](https://github.com/sgl-project/sglang-omni) avec des optimisations pour le modele **Fish Audio S2-Pro TTS** sur GPU H100.

## Optimisations apportees

| Optimisation | Impact |
|---|---|
| Quantification FP8 | ~7.5 GB VRAM economises |
| KV Cache reduit (0.85 -> 0.35) | ~18 GB KV cache, 47 GB libres |
| Preprocessing sur GPU | 4.52s -> 0.07s (64x plus rapide) |
| Endpoint streaming `/v1/audio/speech/stream` | TTFB 11s -> 83ms |
| Fix race condition streaming | Tokens arrives au client |
| Serialisation Tensor pour ZMQ | Transport des codes inter-stages |
| Fix bug concurrence (rid vide) | Batch 5/5 et 8/8 succes |
| Credits relay x16 | Debit inter-stages ameliore |
| CHUNK_SIZE adaptatif (1 puis 5) | TTFB 284ms -> 83ms |
| Cache encodage audio de reference | Pas de re-encodage du meme wav |
| Pre-chargement vocoder au demarrage | Elimination du cold start |
| Preprocessing concurrent (16 threads) | Batch scaling non-lineaire |
| Event async stream (5ms vs 100ms poll) | Reduction jitter streaming |

## Resultats

| Metrique | Batch 1 | Batch 5 | Batch 8 |
|---|---|---|---|
| TTFB avg | **83 ms** | **268 ms** | **283 ms** |
| Total avg | 4.8 s | 7.7 s | 10.1 s |
| Succes | 1/1 | 5/5 | 8/8 |

## Installation

```bash
# Cloner le repo
git clone https://github.com/urbainze/sglang-omni.git
cd sglang-omni

# Creer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dependances
pip install -e .

# Installer sgl-kernel (requis pour FlashAttention 3)
pip install sgl-kernel
```

## Lancer le serveur

### Commande de base (FP8, port 8000)

```bash
python -m sglang_omni.cli.cli serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --quantization fp8 \
  --port 8000
```

Le serveur demarre les 3 stages du pipeline (preprocessing, tts_engine, vocoder) et expose une API compatible OpenAI sur le port 8000.

### Options utiles

| Option | Description | Defaut |
|---|---|---|
| `--quantization fp8` | Quantification FP8 pour reduire la VRAM | Aucune (bfloat16) |
| `--port 8000` | Port du serveur HTTP | 8000 |
| `--model-path` | Chemin ou ID HuggingFace du modele | requis |
| `--config` | Fichier de config YAML du pipeline | requis |

### Lancer en arriere-plan

```bash
nohup python -m sglang_omni.cli.cli serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --quantization fp8 \
  --port 8000 > /tmp/s2pro.log 2>&1 &

# Suivre les logs
tail -f /tmp/s2pro.log
```

Attendre que le serveur affiche `Application startup complete` avant d'envoyer des requetes.

## Tester le serveur

### 1. Test rapide (endpoint standard, sans streaming)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bonjour, ceci est un test.",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": "/chemin/vers/reference.wav",
    "ref_text": ""
  }' \
  --output test_output.wav
```

### 2. Test streaming (TTFB reduit)

```bash
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bonjour, ceci est un test de streaming audio.",
    "voice": "default",
    "response_format": "wav",
    "ref_audio": "/chemin/vers/reference.wav",
    "ref_text": ""
  }' \
  --output test_stream.pcm
```

Le streaming renvoie du PCM 16-bit 44100Hz. Pour convertir en WAV :

```bash
ffmpeg -f s16le -ar 44100 -ac 1 -i test_stream.pcm test_stream.wav
```

### 3. Test Python avec mesure du TTFB

```python
import asyncio, time, httpx

async def test():
    payload = {
        "input": "Bonjour, comment allez-vous aujourd'hui?",
        "voice": "default",
        "response_format": "wav",
        "ref_audio": "/chemin/vers/reference.wav",
        "ref_text": "",
    }
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        async with client.stream("POST", "http://localhost:8000/v1/audio/speech/stream",
                                  json=payload, timeout=60) as resp:
            i = 0
            async for chunk in resp.aiter_bytes():
                t = (time.perf_counter() - t0) * 1000
                if i == 0:
                    print(f"TTFB: {t:.0f}ms")
                print(f"  chunk {i}: {t:.0f}ms, {len(chunk)} bytes")
                i += 1
        print(f"Total: {(time.perf_counter()-t0)*1000:.0f}ms")

asyncio.run(test())
```

### 4. Benchmark batch (TTFB + concurrence)

Le script `benchmarks/bench_stream_ttfb.py` lance des tests en batch 1, 5 et 8 requetes simultanees avec mesure du TTFB :

```bash
python benchmarks/bench_stream_ttfb.py
```

> **Note** : modifier les variables `URL`, `REF_AUDIO` et `TEXT` dans le script selon votre configuration.

## Parametres de l'API

### POST `/v1/audio/speech`

Endpoint standard (attend la fin de la generation avant de repondre).

### POST `/v1/audio/speech/stream`

Endpoint streaming (envoie l'audio par chunks des les premiers tokens).

| Parametre | Type | Description |
|---|---|---|
| `input` | string | Texte a synthetiser |
| `voice` | string | Voix (utiliser `"default"`) |
| `response_format` | string | Format (`"wav"`) |
| `ref_audio` | string | Chemin vers l'audio de reference (voice cloning) |
| `ref_text` | string | Transcription de l'audio de reference (optionnel) |

## Fichiers modifies (par rapport a l'upstream)

| Fichier | Modification |
|---|---|
| `sglang_omni/config/compiler.py` | Injection automatique du parametre quantization |
| `sglang_omni/models/fishaudio_s2_pro/config.py` | Ajout champ quantization, credits relay x16 |
| `sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py` | GPU codec, quantization, mem_fraction=0.35, cache ref audio |
| `sglang_omni/models/fishaudio_s2_pro/pipeline/engine_io.py` | Fix rid vide pour la concurrence |
| `sglang_omni/executors/engine_executor.py` | Race condition stream + serialisation tensor |
| `sglang_omni/executors/preprocessing_executor.py` | Thread pool dedie (16 workers) pour preprocessing concurrent |
| `sglang_omni/serve/openai_api.py` | Streaming endpoint, CHUNK_SIZE adaptatif, pre-warm vocoder |
| `sglang_omni/pipeline/coordinator.py` | Debug logging stream |
| `sglang_omni/pipeline/worker/runtime.py` | Event async stream (5ms), debug logging |

## Credits

- [sglang-omni](https://github.com/sgl-project/sglang-omni) - Framework original
- [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) - Modele TTS
- [SGLang](https://github.com/sgl-project/sglang) - Backend LLM serving
