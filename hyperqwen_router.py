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

# Reasoning-effort policy (ROADMAP 2026-10-31, items 37-39).
#   v1      caller effort passes through, mapped to the template's levels;
#           absent -> medium (EFFORT_DEFAULT)
#   legacy  force xhigh everywhere (pre-2026-10-31 behavior; rollback)
EFFORT_POLICY = os.environ.get("QWEN_ROUTER_EFFORT_POLICY", "v1")
EFFORT_DEFAULT = "medium"
# vLLM 0.32 protocol vocabulary -> the served template's levels
# (xhigh/medium/low) plus none (= thinking off, engine derives
# enable_thinking=false itself).
EFFORT_MAP = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}
# Injected default for requests that set neither max_tokens nor
# max_completion_tokens (chat) nor max_output_tokens (Responses). Blast-radius
# guard (ROADMAP item 15 / incident d309a671); callers who set an explicit cap
# are unaffected up to the ceiling.
MAX_TOKENS_DEFAULT = int(os.environ.get("QWEN_ROUTER_MAX_TOKENS", 16384))
# Hard ceiling for explicit Responses max_output_tokens values. Chat keeps no
# ceiling (legacy behavior, matches Bifrost's 32K advertised max_output_tokens);
# Responses traffic clamps here because its no-default fallback is the whole
# remaining window.
MAX_TOKENS_CEILING = int(os.environ.get("QWEN_ROUTER_MAX_TOKENS_CEILING", 32768))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [router] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router")

MODELS_RESPONSE = json.dumps({
    "object": "list",
    "data": [
        {"id": "qwen-gpu0", "object": "model", "owned_by": "black-monolith-gpu0-x4", "max_model_len": 200000, "permission": []},
        {"id": "qwen-gpu2", "object": "model", "owned_by": "black-monolith-gpu2-x16", "max_model_len": 200000, "permission": []},
    ]
}).encode("utf-8")


def target_for_model(req_model: str) -> int:
    """Map public model aliases to the two maintained SuperQwen engines."""
    model = req_model.lower()
    if "gpu2" in model or "fast" in model:
        return PORT_GPU2
    return PORT_GPU0


def fallback_for_target(_target_port: int) -> None:
    """The consolidated deployment has no alternate backends to fall through to."""
    return None


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
        is_responses_api = path.startswith("/v1/responses")
        if body:
            try:
                j = json.loads(body.decode("utf-8"))
                req_model = j.get("model", "")
                model_name = req_model
                target_port = target_for_model(req_model)
                j["model"] = "qwen3.8-27b"
                if j.get("repetition_penalty") is None:
                    j["repetition_penalty"] = 1.08
                if is_responses_api:
                    # Responses API: the field is max_output_tokens. The
                    # get_max_tokens fallback when it is absent is
                    # max_model_len - input_length (up to ~200K), which is the
                    # d309a671 runaway shape; apply the same blast-radius
                    # default as chat, and a ceiling on explicit values above
                    # it. Callers who need more than the ceiling must raise
                    # QWEN_ROUTER_MAX_TOKENS.
                    if j.get("max_output_tokens") is None:
                        j["max_output_tokens"] = MAX_TOKENS_DEFAULT
                    elif j["max_output_tokens"] > MAX_TOKENS_CEILING:
                        logger.warning("responses max_output_tokens %d > ceiling %d; clamping",
                                       j["max_output_tokens"], MAX_TOKENS_CEILING)
                        j["max_output_tokens"] = MAX_TOKENS_CEILING
                elif "max_tokens" not in j and "max_completion_tokens" not in j:
                    j["max_tokens"] = MAX_TOKENS_DEFAULT
                if EFFORT_POLICY == "v1":
                    # Passthrough with validation + nearest-level mapping.
                    # The served template accepts only xhigh/medium/low and
                    # raises on anything else; vLLM 0.32's protocol accepts the
                    # OpenAI vocabulary (none/minimal/low/medium/high/xhigh/max),
                    # so map that vocabulary onto the template's three levels
                    # (plus none = thinking off) instead of forwarding unknown
                    # values into the template's raise path. Absent -> medium.
                    # Precedence: chat_template_kwargs > top-level > reasoning.effort
                    # > params.reasoning.effort (mirrors the old overwrite order).
                    raw_effort = None
                    raw_source = None
                    ctk = j.get("chat_template_kwargs")
                    if isinstance(ctk, dict) and "reasoning_effort" in ctk:
                        raw_effort, raw_source = ctk.get("reasoning_effort"), "ctk"
                    elif j.get("reasoning_effort") is not None:
                        raw_effort, raw_source = j.get("reasoning_effort"), "top"
                    elif isinstance(j.get("reasoning"), dict) and j["reasoning"].get("effort") is not None:
                        raw_effort, raw_source = j["reasoning"].get("effort"), "reasoning"
                    elif isinstance(j.get("params"), dict) and isinstance(j["params"].get("reasoning"), dict) \
                            and j["params"]["reasoning"].get("effort") is not None:
                        raw_effort, raw_source = j["params"]["reasoning"].get("effort"), "params"
                    mapped = EFFORT_MAP.get(str(raw_effort).lower()) if raw_effort is not None else None
                    if mapped is None:
                        if raw_effort is not None:
                            logger.warning("effort %r from %s not recognized; mapping to %s",
                                           raw_effort, raw_source, EFFORT_DEFAULT)
                        mapped = EFFORT_DEFAULT
                    j["reasoning_effort"] = mapped
                    if isinstance(ctk, dict):
                        ctk["reasoning_effort"] = mapped
                    if isinstance(j.get("reasoning"), dict) and "effort" in j["reasoning"]:
                        j["reasoning"]["effort"] = mapped
                    if isinstance(j.get("params"), dict) and isinstance(j["params"].get("reasoning"), dict) \
                            and "effort" in j["params"]["reasoning"]:
                        j["params"]["reasoning"]["effort"] = mapped
                    logger.info("effort: requested=%r (%s) -> %s", raw_effort, raw_source, mapped)
                else:
                    # Legacy policy (rollback): force xhigh everywhere.
                    j["reasoning_effort"] = "xhigh"
                    if isinstance(j.get("chat_template_kwargs"), dict):
                        if "reasoning_effort" in j["chat_template_kwargs"]:
                            j["chat_template_kwargs"]["reasoning_effort"] = "xhigh"
                    if isinstance(j.get("reasoning"), dict):
                        if "effort" in j["reasoning"]:
                            j["reasoning"]["effort"] = "xhigh"
                    if isinstance(j.get("params"), dict) and isinstance(j["params"].get("reasoning"), dict):
                        if "effort" in j["params"]["reasoning"]:
                            j["params"]["reasoning"]["effort"] = "xhigh"
                # Qwen chat template requires the system message first. Hermes injects
                # mid-conversation system messages (out-of-band steering, tool preambles);
                # hoist every system message to the front preserving their relative order.
                msgs = j.get("messages")
                if isinstance(msgs, list):
                    if any(m.get("role") == "system" for m in msgs):
                        systems = [m for m in msgs if m.get("role") == "system"]
                        others = [m for m in msgs if m.get("role") != "system"]
                        # Template allows exactly ONE system message at index 0; merge all
                        # system messages (Bifrost appends its own) into a single block.
                        merged = {"role": "system", "content": "\n\n".join(
                            m.get("content") or "" for m in systems if isinstance(m.get("content"), str)
                        )}
                        j["messages"] = [merged] + others
                    logger.info("req roles: %s", [m.get("role") for m in j["messages"]])
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
            fallback_port = fallback_for_target(target_port)
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
