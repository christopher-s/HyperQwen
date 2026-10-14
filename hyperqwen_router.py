#!/usr/bin/env python3
"""
HyperQwen Dual-3090 Model Router.
Listens on 0.0.0.0:8020 and dispatches requests:
  - 'qwen-3.8-27b-fast' or 'qwen3.8-27b-fast' -> 127.0.0.1:18021 (PCIe x16 GPU2: MTP 180K)
  - 'qwen-3.8-27b' or 'qwen3.8-27b'           -> 127.0.0.1:18020 (PCIe x4 GPU0: KVarN 180K 2-stream)
Streams SSE responses and rewrites model identifiers seamlessly.
"""

import asyncio
import json
import logging
import os
import signal
import sys

PORT_GPU0 = int(os.environ.get("PORT_GPU0", 18020))
PORT_GPU0_ALT = int(os.environ.get("PORT_GPU0_ALT", 18024))
PORT_GPU2 = int(os.environ.get("PORT_GPU2", 18021))
PORT_GPU2_ALT = int(os.environ.get("PORT_GPU2_ALT", 18022))
LISTEN_PORT = int(os.environ.get("PORT_ROUTER", 8020))
LISTEN_HOST = os.environ.get("HOST_ROUTER", "0.0.0.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [router] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router")

MODELS_RESPONSE = json.dumps({
    "object": "list",
    "data": [
        {"id": "qwen-gpu0", "object": "model", "owned_by": "black-monolith-gpu0-x4", "max_model_len": 196608, "permission": []},
        {"id": "qwen-gpu2", "object": "model", "owned_by": "black-monolith-gpu2-x16", "max_model_len": 196608, "permission": []},
        {"id": "qwen-dual", "object": "model", "owned_by": "black-monolith-dual-3090", "max_model_len": 262144, "permission": []},
        {"id": "qwen-3.8-27b", "object": "model", "owned_by": "hyperqwen-gpu0-x4", "max_model_len": 180000, "permission": []},
        {"id": "qwen-3.8-27b-fast", "object": "model", "owned_by": "hyperqwen-gpu2-x16", "max_model_len": 180000, "permission": []},
    ]
}).encode("utf-8")


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    backend_reader = None
    backend_writer = None
    try:
        req_line = await reader.readline()
        if not req_line:
            writer.close()
            return
        parts = req_line.decode("utf-8", "replace").split()
        if len(parts) < 2:
            writer.close()
            return
        method, path = parts[0], parts[1]

        headers: dict[str, str] = {}
        while True:
            hline = await reader.readline()
            if not hline or hline in (b"\r\n", b"\n"):
                break
            hstr = hline.decode("utf-8", "replace").strip()
            if ":" in hstr:
                k, v = hstr.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        content_length = int(headers.get("content-length", 0))
        body = b""
        if content_length > 0:
            body = await reader.readexactly(content_length)

        # Health endpoint
        if method == "GET" and path in ("/health", "/health/"):
            resp = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}"
            writer.write(resp)
            await writer.drain()
            writer.close()
            return

        # Models endpoint
        if method == "GET" and path.startswith("/v1/models"):
            resp = (
                f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(MODELS_RESPONSE)}\r\nConnection: close\r\n\r\n".encode("utf-8")
                + MODELS_RESPONSE
            )
            writer.write(resp)
            await writer.drain()
            writer.close()
            return

        # Target selection based on model name
        target_port = PORT_GPU0
        model_name = "qwen-3.8-27b"
        if body:
            try:
                j = json.loads(body.decode("utf-8"))
                req_model = j.get("model", "")
                model_name = req_model
                if "gpu2" in req_model or "fast-alt" in req_model:
                    target_port = PORT_GPU2_ALT
                    j["model"] = "qwen-gpu2"
                elif "gpu0" in req_model:
                    target_port = PORT_GPU0_ALT
                    j["model"] = "qwen-gpu0"
                elif "fast" in req_model:
                    target_port = PORT_GPU2
                    j["model"] = "qwen3.8-27b"
                elif "dual" in req_model:
                    target_port = PORT_GPU0
                    j["model"] = "qwen3.8-27b"
                else:
                    target_port = PORT_GPU0
                    j["model"] = "qwen3.8-27b"
                if j.get("repetition_penalty") is None:
                    j["repetition_penalty"] = 1.08
                if "max_tokens" not in j and "max_completion_tokens" not in j:
                    j["max_tokens"] = 8192
                # Lock reasoning effort to medium: Qwen template strictly requires xhigh/medium/low,
                # rejecting generic "high" or "max", and medium prevents runaway deliberation loops.
                j["reasoning_effort"] = "medium"
                if isinstance(j.get("chat_template_kwargs"), dict):
                    if "reasoning_effort" in j["chat_template_kwargs"]:
                        j["chat_template_kwargs"]["reasoning_effort"] = "medium"
                # Qwen chat template requires the system message first. Hermes injects
                # mid-conversation system messages (out-of-band steering, tool preambles);
                # hoist every system message to the front preserving their relative order.
                msgs = j.get("messages")
                if isinstance(msgs, list) and any(m.get("role") == "system" for m in msgs):
                    systems = [m for m in msgs if m.get("role") == "system"]
                    others = [m for m in msgs if m.get("role") != "system"]
                    j["messages"] = systems + others
                body = json.dumps(j).encode("utf-8")
            except Exception as e:
                logger.warning("Error parsing JSON body: %s", e)

        # Connect to backend
        try:
            backend_reader, backend_writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", target_port),
                timeout=10.0,
            )
        except Exception as e:
            fallback_port = None
            if target_port == PORT_GPU2:
                fallback_port = PORT_GPU2_ALT
            elif target_port == PORT_GPU2_ALT:
                fallback_port = PORT_GPU2
            elif target_port == PORT_GPU0:
                fallback_port = PORT_GPU0_ALT
            elif target_port == PORT_GPU0_ALT:
                fallback_port = PORT_GPU0
            if fallback_port is not None:
                try:
                    logger.info("Target port %d unavailable, attempting fallback port %d", target_port, fallback_port)
                    backend_reader, backend_writer = await asyncio.wait_for(
                        asyncio.open_connection("127.0.0.1", fallback_port),
                        timeout=5.0,
                    )
                    target_port = fallback_port
                except Exception:
                    backend_writer = None
            if backend_writer is None or backend_reader is None:
                logger.error("Failed to connect to backend on port %d: %s", target_port, e)
                err_msg = json.dumps({"error": f"Backend on port {target_port} unavailable: {e}"}).encode("utf-8")
                resp = f"HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: {len(err_msg)}\r\nConnection: close\r\n\r\n".encode("utf-8") + err_msg
                writer.write(resp)
                await writer.drain()
                writer.close()
                return

        # Send request line & headers to backend
        backend_writer.write(f"{method} {path} HTTP/1.1\r\n".encode("utf-8"))
        for hk, hv in headers.items():
            if hk in ("content-length", "host"):
                continue
            backend_writer.write(f"{hk}: {hv}\r\n".encode("utf-8"))
        backend_writer.write(f"host: 127.0.0.1:{target_port}\r\n".encode("utf-8"))
        if body:
            backend_writer.write(f"content-length: {len(body)}\r\n\r\n".encode("utf-8"))
            backend_writer.write(body)
        else:
            backend_writer.write(b"\r\n")
        await backend_writer.drain()

        # Stream response back
        while True:
            chunk = await backend_reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()

    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        logger.error("Unhandled proxy exception: %s", e)
    finally:
        if backend_writer is not None:
            backend_writer.close()
        writer.close()


async def main() -> None:
    server = await asyncio.start_server(handle_client, LISTEN_HOST, LISTEN_PORT, reuse_address=True)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("Serving HyperQwen router on %s (GPU0->%d, GPU2->%d)", addrs, PORT_GPU0, PORT_GPU2)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _sig_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except NotImplementedError:
            pass

    async with server:
        await stop_event.wait()
        logger.info("Closing router server...")


if __name__ == "__main__":
    asyncio.run(main())
