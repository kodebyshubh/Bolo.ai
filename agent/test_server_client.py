"""
Test client for agent/server.py's WebSocket endpoint: sends a complete
input WAV file's bytes, receives and saves each sentence's audio as it
arrives, prints timing. Needs the real server running (with a GPU) --
see agent/server.py's usage comment.

Usage:
    python agent/test_server_client.py path/to/input.wav ws://localhost:8000/ws/converse
"""
import asyncio
import json
import sys

import websockets


async def run(audio_path, uri):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(audio_bytes)

        pending_metadata = None
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if pending_metadata is None:
                    print("warning: received audio bytes with no preceding metadata")
                    continue
                index = pending_metadata["index"]
                path = f"received_sentence{index}.wav"
                with open(path, "wb") as f:
                    f.write(message)
                print(f"  wrote {path} ({len(message)} bytes)")
                pending_metadata = None
            else:
                data = json.loads(message)
                if data["type"] == "sentence_audio":
                    print(f"[sentence {data['index']}]: {data['sentence']}")
                    pending_metadata = data
                elif data["type"] == "done":
                    print(f"\n[full reply]: {data['reply_text']}")
                    print("timings:", data["timings"])
                    return
                elif data["type"] == "error":
                    print(f"[error]: {data['message']}")
                    return


if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/audio/selfrecorded/selfrecorded_011.wav"
    uri = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/ws/converse"
    asyncio.run(run(audio_path, uri))
