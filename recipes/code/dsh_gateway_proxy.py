# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Expose only authenticated, expiring session model routes to remote sandboxes.

Run behind a TLS tunnel on the same node as the recipe runner. The private route
folder is shared with that runner; it is never a public configuration endpoint.
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse

_SESSION = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SESSION_PATH = re.compile(r"/sessions/([A-Za-z0-9_-]{1,128})/v1\Z")
_MAX_TTL_SECONDS = 86400


def _parse_url(value):
    try:
        parsed = urlsplit(value)
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError
        # Accessing port validates malformed or out-of-range port numbers.
        _ = parsed.port
        if parsed.query or parsed.fragment or "?" in value or "#" in value:
            raise ValueError
        if any(character.isspace() for character in value):
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Expected an absolute URL without credentials, query, or fragment") from exc
    return parsed


def _session_url(value):
    parsed = _parse_url(value)
    match = _SESSION_PATH.fullmatch(parsed.path)
    if parsed.scheme not in {"http", "https"} or match is None:
        raise ValueError("Gateway URL must be http(s)://host/sessions/<id>/v1")
    return match.group(1)


def _route_path(route_dir, session_id):
    return Path(route_dir) / (hashlib.sha256(session_id.encode()).hexdigest() + ".json")


@contextmanager
def register_route(gateway_url, public_origin, route_dir, ttl_seconds=3600):
    """Yield ``(public_session_base_url, bearer_token)``; revoke on every exit.

    TTL must be finite and between 1 second and 24 hours. Registrations cannot
    overwrite another live file for the same session. Callers must not log the
    returned token, which is intended only for their isolated sandbox process.
    """
    session_id = _session_url(gateway_url)
    public = _parse_url(public_origin)
    if public.scheme != "https" or public.path:
        raise ValueError("Public URL must be an HTTPS origin with no path")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int | float)
        or not math.isfinite(ttl_seconds)
        or not 1 <= ttl_seconds <= _MAX_TTL_SECONDS
    ):
        raise ValueError("Route lifetime must be between 1 and 86400 seconds")

    directory = Path(route_dir)
    if directory.is_symlink():
        raise ValueError("Route directory must not be a symlink")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = _route_path(directory, session_id)
    token = secrets.token_urlsafe(32)
    record = {"upstream_base_url": gateway_url, "token": token, "expires_at": time.time() + ttl_seconds}
    # Write privately, then publish atomically without replacing another route.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".route-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    try:
        yield public_origin + f"/sessions/{session_id}/v1", token
    finally:
        path.unlink(missing_ok=True)


def _authorized_route(route_dir, session_id, authorization):
    if not _SESSION.fullmatch(session_id) or not authorization.startswith("Bearer "):
        return None
    supplied_token = authorization[7:]
    try:
        descriptor = os.open(_route_path(route_dir, session_id), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor) as handle:
            record = json.load(handle)
        token = record["token"]
        expires_at = record["expires_at"]
        if not isinstance(token, str) or not isinstance(expires_at, int | float):
            return None
        if not math.isfinite(expires_at) or expires_at <= time.time():
            return None
        if not hmac.compare_digest(supplied_token.encode(), token.encode()):
            return None
        if _session_url(record["upstream_base_url"]) != session_id:
            return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return record


class _RelayResponse(StreamingResponse):
    """Close the upstream even if the client disconnects before body iteration."""

    def __init__(self, upstream, client):
        self.upstream = upstream
        self.client = client
        super().__init__(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers={
                "content-type": upstream.headers.get("content-type", "application/json"),
                "cache-control": "no-store",
                "x-accel-buffering": "no",
            },
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.upstream.aclose()
            finally:
                await self.client.aclose()


class _HeartbeatRelayResponse(StreamingResponse):
    """Public liveness while the origin waits for a complete generation."""

    def __init__(self, client, upstream_request, heartbeat_seconds):
        self.client = client
        self.upstream_request = upstream_request
        self.upstream = None
        self.pending = None
        self.heartbeat_seconds = heartbeat_seconds
        super().__init__(
            self._body(),
            media_type="text/event-stream",
            headers={"cache-control": "no-store", "x-accel-buffering": "no"},
        )

    async def _wait(self, awaitable):
        self.pending = asyncio.ensure_future(awaitable)
        while not self.pending.done():
            done, _ = await asyncio.wait({self.pending}, timeout=self.heartbeat_seconds)
            if not done:
                yield b": keep-alive\n\n"

    async def _body(self):
        yield b": keep-alive\n\n"
        try:
            async for ping in self._wait(self.client.send(self.upstream_request, stream=True)):
                yield ping
            self.upstream = self.pending.result()
            if not 200 <= self.upstream.status_code < 300:
                raise ValueError("upstream status")
            if self.upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                raise ValueError("upstream content type")
            iterator = self.upstream.aiter_bytes().__aiter__()
            buffer = b""
            while True:
                async for ping in self._wait(anext(iterator)):
                    yield ping
                try:
                    chunk = self.pending.result()
                except StopAsyncIteration:
                    if buffer.strip():
                        raise ValueError("unfinished SSE frame") from None
                    return
                buffer += chunk
                if len(buffer) > 16 * 1024**2:
                    raise ValueError("SSE frame too large")
                while match := re.search(rb"\r\n\r\n|\n\n|\r\r", buffer):
                    yield buffer[: match.end()]
                    buffer = buffer[match.end() :]
        except (httpx.HTTPError, ValueError, OSError):
            # The fixed DSH parser ignores error JSON but rejects EOF without DONE.
            # Never fabricate successful termination after a transport failure.
            yield (
                b'event: error\ndata: {"error":{"message":"Policy gateway stream failed",'
                b'"type":"server_error","code":"gateway_stream_error"}}\n\n'
            )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.pending is not None:
                if not self.pending.done():
                    self.pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    result = await self.pending
                    if isinstance(result, httpx.Response) and self.upstream is None:
                        self.upstream = result
            try:
                if self.upstream is not None:
                    await self.upstream.aclose()
            finally:
                await self.client.aclose()


def create_app(route_dir, *, transport=None, heartbeat_seconds=15.0):
    """Create the relay; ``transport`` is injectable for offline contract tests."""
    if not math.isfinite(heartbeat_seconds) or not 0 < heartbeat_seconds <= 30:
        raise ValueError("heartbeat interval must be between 0 and 30 seconds")
    directory = Path(route_dir)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        if request.url.query:
            raise HTTPException(400, "Query parameters are not supported")
        record = _authorized_route(directory, session_id, request.headers.get("authorization", ""))
        if record is None:
            raise HTTPException(401, "Invalid or expired session authorization")
        client = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(300.0, connect=10.0, write=30.0, pool=10.0),
        )
        upstream = None
        try:
            upstream_request = client.build_request(
                "POST",
                record["upstream_base_url"] + "/chat/completions",
                content=await request.body(),
                headers={"content-type": "application/json", "accept": "application/json, text/event-stream"},
            )
            try:
                payload = json.loads(upstream_request.content)
            except (ValueError, UnicodeError):
                payload = None
            if isinstance(payload, dict) and payload.get("stream") is True:
                return _HeartbeatRelayResponse(client, upstream_request, heartbeat_seconds)
            upstream = await client.send(upstream_request, stream=True)
            if not 200 <= upstream.status_code < 300:
                raise HTTPException(502, "Policy gateway returned an unsuccessful response")
            return _RelayResponse(upstream, client)
        except BaseException as exc:
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                await client.aclose()
            if isinstance(exc, httpx.TimeoutException):
                raise HTTPException(504, "Policy gateway timed out") from None
            if isinstance(exc, httpx.HTTPError):
                raise HTTPException(502, "Policy gateway could not be reached") from None
            raise

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.route_dir), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
