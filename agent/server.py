"""
PRD v2 step 8: WebSocket transport exposing agent.pipeline.run_streaming()
over the network, so a browser client can receive and play each sentence's
audio as soon as it's ready, instead of waiting for one full HTTP response.

Protocol on /ws/converse:
  client -> server: one binary WebSocket message containing a complete
    input WAV file's bytes (the client records a full utterance first --
    true incremental mic-chunk upload to the server is a larger separate
    feature, not attempted in this step).
  server -> client, once, as soon as transcription completes:
    a text (JSON) frame: {"type": "stt", "text": str, "language": str}
  server -> client, per sentence, in order:
    1. a text (JSON) frame: {"type": "sentence_audio", "index": int, "sentence": str}
    2. a binary frame: that sentence's audio, WAV-encoded
  server -> client, once at the end:
    a text (JSON) frame: {"type": "done", "reply_text": str, "timings": dict}
  server -> client, on error:
    a text (JSON) frame: {"type": "error", "message": str}

Needs a GPU (for TTS, via agent.pipeline) and the API keys agent.pipeline
itself needs (GROQ_API_KEY, ASSEMBLYAI_API_KEY).

Known scope limitation, stated rather than hidden: run_streaming() is a
blocking generator (its internal queue waits aren't async-aware), so
consuming it directly inside this async handler blocks FastAPI's event
loop for the duration of one conversation turn. Fine for a single-
connection portfolio demo (one user talking to the agent at a time, which
is the actual scope here); a real multi-user deployment would need to run
run_streaming() in a thread executor (e.g. asyncio.to_thread per event)
so other connections aren't blocked while one is being served.

Usage:
    uvicorn agent.server:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

load_dotenv()

from agent.pipeline import run_streaming
from agent.tts import SAMPLE_RATE

app = FastAPI()


def _audio_to_wav_bytes(audio, sample_rate):
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


@app.websocket("/ws/converse")
async def converse(websocket: WebSocket):
    await websocket.accept()
    try:
        audio_bytes = await websocket.receive_bytes()

        for event in run_streaming(audio_bytes):
            if event["type"] == "stt":
                await websocket.send_json({
                    "type": "stt",
                    "text": event["text"],
                    "language": event["language"],
                })
            elif event["type"] == "sentence_audio":
                await websocket.send_json({
                    "type": "sentence_audio",
                    "index": event["index"],
                    "sentence": event["sentence"],
                })
                wav_bytes = _audio_to_wav_bytes(event["audio"], SAMPLE_RATE)
                await websocket.send_bytes(wav_bytes)
            elif event["type"] == "done":
                await websocket.send_json({
                    "type": "done",
                    "reply_text": event["reply_text"],
                    "timings": event["timings"],
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
