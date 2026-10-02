# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""CPU contract checks for the public, session-scoped model relay."""

import asyncio
import hashlib
import json
import stat
import time

import httpx
import pytest

from recipes.code.dsh_gateway_proxy import create_app, register_route

SESSION = "episode_123-ab"
UPSTREAM = f"http://127.0.0.1:8765/sessions/{SESSION}/v1"
PUBLIC = "https://policy.example.org"


def route_path(directory):
    return directory / (hashlib.sha256(SESSION.encode()).hexdigest() + ".json")


def test_route_is_private_preserves_session_and_revokes_after_failure(tmp_path):
    directory = tmp_path / "routes"
    with pytest.raises(RuntimeError, match="agent failed"):
        with register_route(UPSTREAM, PUBLIC, directory, 60) as (url, token):
            assert url == f"{PUBLIC}/sessions/{SESSION}/v1"
            assert len(token) >= 32
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            assert stat.S_IMODE(route_path(directory).stat().st_mode) == 0o600
            record = json.loads(route_path(directory).read_text())
            assert record["upstream_base_url"] == UPSTREAM
            assert record["token"] == token
            assert time.time() < record["expires_at"] <= time.time() + 60
            raise RuntimeError("agent failed")
    assert not list(directory.iterdir())


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.org",
        "https://u:p@example.org",
        "https://example.org/x",
        "https://example.org?q=x",
        "https://example.org#x",
    ],
)
def test_rejects_non_origin_public_url(tmp_path, origin):
    with pytest.raises(ValueError):
        with register_route(UPSTREAM, origin, tmp_path, 60):
            pass


@pytest.mark.parametrize(
    "upstream",
    [
        "http://localhost/v1",
        "http://localhost/sessions/../v1",
        "http://u:p@localhost/sessions/abc/v1",
        "http://localhost/sessions/abc/v1?x=1",
        "ftp://localhost/sessions/abc/v1",
    ],
)
def test_rejects_unsafe_or_non_session_upstream(tmp_path, upstream):
    with pytest.raises(ValueError):
        with register_route(upstream, PUBLIC, tmp_path, 60):
            pass


@pytest.mark.parametrize("ttl", [0, -1, 86401, float("inf"), float("nan"), True])
def test_rejects_unbounded_lifetime(tmp_path, ttl):
    with pytest.raises(ValueError):
        with register_route(UPSTREAM, PUBLIC, tmp_path, ttl):
            pass


def test_live_session_cannot_be_overwritten(tmp_path):
    with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (_, token):
        with pytest.raises(FileExistsError):
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60):
                pass
        assert json.loads(route_path(tmp_path).read_text())["token"] == token


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[]}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


def test_stream_forwarding_and_access_boundaries(tmp_path):
    stream = TrackedStream()
    requests = []

    async def upstream(request):
        requests.append(request)
        assert str(request.url) == UPSTREAM + "/chat/completions"
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {"model": "policy", "stream": True}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async def run():
        app = create_app(tmp_path, transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (base, token):
                endpoint = base + "/chat/completions"
                for headers in ({}, {"Authorization": "Bearer wrong"}):
                    assert (await client.post(endpoint, headers=headers, json={})).status_code == 401
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.post(endpoint, headers=headers, json={"model": "policy", "stream": True})
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                assert response.content.endswith(b"data: [DONE]\n\n")
                assert stream.closed
                assert (await client.post(f"/sessions/{SESSION}/reward", headers=headers, json={})).status_code == 404
                assert (await client.post("/v1/chat/completions", headers=headers, json={})).status_code == 404
                assert (await client.get(endpoint, headers=headers)).status_code == 405
                assert (
                    await client.post(endpoint + "?upstream=http://evil", headers=headers, json={})
                ).status_code == 400
            assert (await client.post(endpoint, headers=headers, json={})).status_code == 401
        assert len(requests) == 1

    asyncio.run(run())


def test_expired_and_corrupt_routes_fail_closed(tmp_path):
    async def unexpected(request):
        pytest.fail("Unauthorized request reached upstream")

    async def run():
        app = create_app(tmp_path, transport=httpx.MockTransport(unexpected))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (base, token):
                record = json.loads(route_path(tmp_path).read_text())
                record["expires_at"] = time.time() - 1
                route_path(tmp_path).write_text(json.dumps(record))
                headers = {"Authorization": f"Bearer {token}"}
                assert (await client.post(base + "/chat/completions", headers=headers, json={})).status_code == 401
                route_path(tmp_path).write_text("broken")
                assert (await client.post(base + "/chat/completions", headers=headers, json={})).status_code == 401

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["http", "timeout", "connection"])
def test_upstream_errors_are_sanitized(tmp_path, failure):
    secret = "private-upstream-secret"

    async def upstream(request):
        if failure == "timeout":
            raise httpx.ReadTimeout(secret)
        if failure == "connection":
            raise httpx.ConnectError(secret)
        return httpx.Response(500, text=secret)

    async def run():
        app = create_app(tmp_path, transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (base, token):
                response = await client.post(
                    base + "/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={}
                )
                assert response.status_code == (504 if failure == "timeout" else 502)
                assert secret not in response.text
                assert token not in response.text

    asyncio.run(run())


def test_client_disconnect_closes_stream_and_transport(tmp_path):
    class WaitingStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    class TrackedTransport(httpx.AsyncBaseTransport):
        closed = False

        def __init__(self):
            self.stream = WaitingStream()

        async def handle_async_request(self, request):
            return httpx.Response(200, stream=self.stream)

        async def aclose(self):
            self.closed = True

    async def run():
        transport = TrackedTransport()
        app = create_app(tmp_path, transport=transport)
        first_chunk = asyncio.Event()
        body_received = False

        async def receive():
            nonlocal body_received
            if not body_received:
                body_received = True
                return {"type": "http.request", "body": b"{}", "more_body": False}
            await first_chunk.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_chunk.set()

        with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (_, token):
            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.0"},
                "method": "POST",
                "scheme": "https",
                "path": f"/sessions/{SESSION}/v1/chat/completions",
                "query_string": b"",
                "headers": [(b"authorization", f"Bearer {token}".encode())],
                "server": ("policy.example.org", 443),
                "client": ("127.0.0.1", 10000),
                "root_path": "",
                "http_version": "1.1",
            }
            await asyncio.wait_for(app(scope, receive, send), timeout=2)
        assert first_chunk.is_set()
        assert transport.stream.closed
        assert transport.closed

    asyncio.run(run())


def test_stream_timeout_closes_resources(tmp_path):
    class TimeoutStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"data: first\n\n"
            raise httpx.ReadTimeout("upstream stream stalled")

        async def aclose(self):
            self.closed = True

    class TrackedTransport(httpx.AsyncBaseTransport):
        closed = False

        def __init__(self):
            self.stream = TimeoutStream()

        async def handle_async_request(self, request):
            return httpx.Response(200, stream=self.stream)

        async def aclose(self):
            self.closed = True

    async def run():
        transport = TrackedTransport()
        app = create_app(tmp_path, transport=transport)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (base, token):
                # Headers have already been sent, so a mid-stream failure must
                # terminate the response; it cannot invent a successful [DONE].
                with pytest.raises(httpx.ReadTimeout):
                    await client.post(base + "/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={})
        assert transport.stream.closed
        assert transport.closed

    asyncio.run(run())


async def _stream_asgi(app, token, send, disconnect=None):
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}
        await (disconnect or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "POST",
        "scheme": "https",
        "path": f"/sessions/{SESSION}/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
        "server": ("policy.example.org", 443),
        "client": ("127.0.0.1", 10000),
        "root_path": "",
        "http_version": "1.1",
    }
    await asyncio.wait_for(app(scope, receive, send), 2)


def test_heartbeat_before_headers_and_between_partial_frames(tmp_path):
    async def run():
        header_release, body_release = asyncio.Event(), asyncio.Event()
        messages = []
        headers_pending = True
        partial_sent = False

        class Frames(httpx.AsyncByteStream):
            async def __aiter__(self):
                nonlocal partial_sent
                yield b'data: {"choices":'
                partial_sent = True
                await body_release.wait()
                yield b"[]}\n\ndata: [DONE]\n\n"

        async def upstream(_):
            nonlocal headers_pending
            await header_release.wait()
            headers_pending = False
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Frames())

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                assert headers_pending
            if message.get("body") == b": keep-alive\n\n":
                if not header_release.is_set():
                    header_release.set()
                elif partial_sent:
                    body_release.set()

        app = create_app(tmp_path, transport=httpx.MockTransport(upstream), heartbeat_seconds=0.005)
        with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (_, token):
            await _stream_asgi(app, token, send)
        body = b"".join(m.get("body", b"") for m in messages)
        assert body.count(b": keep-alive\n\n") >= 2
        assert body.replace(b": keep-alive\n\n", b"") == b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        assert body_release.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["status", "timeout", "content_type", "body_timeout"])
def test_stream_failure_has_sanitized_error_and_no_done(tmp_path, failure):
    class FailingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadTimeout("private-error-details")

    async def upstream(_):
        if failure == "timeout":
            raise httpx.ReadTimeout("private-error-details")
        if failure == "status":
            return httpx.Response(503, text="private-error-details")
        if failure == "content_type":
            return httpx.Response(200, json={"private": "error-details"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=FailingBody())

    async def run():
        app = create_app(tmp_path, transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (base, token):
                response = await client.post(
                    base + "/chat/completions", json={"stream": True}, headers={"Authorization": f"Bearer {token}"}
                )
        assert response.status_code == 200
        assert b"event: error\n" in response.content
        assert b"private" not in response.content and b"[DONE]" not in response.content

    asyncio.run(run())


def test_disconnect_while_waiting_headers_cancels_upstream(tmp_path):
    async def run():
        started, cancelled, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Transport(httpx.AsyncBaseTransport):
            closed = False

            async def handle_async_request(self, _):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            async def aclose(self):
                self.closed = True

        transport = Transport()

        async def send(message):
            if message.get("body") == b": keep-alive\n\n" and started.is_set():
                disconnected.set()

        app = create_app(tmp_path, transport=transport, heartbeat_seconds=0.005)
        with register_route(UPSTREAM, PUBLIC, tmp_path, 60) as (_, token):
            await _stream_asgi(app, token, send, disconnected)
        assert cancelled.is_set() and transport.closed

    asyncio.run(run())


def test_unauthorized_stream_never_sends_heartbeat(tmp_path):
    async def upstream(_):
        pytest.fail("unauthorized upstream")

    async def run():
        app = create_app(tmp_path, transport=httpx.MockTransport(upstream))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PUBLIC) as client:
            response = await client.post(f"/sessions/{SESSION}/v1/chat/completions", json={"stream": True})
        assert response.status_code == 401 and b"keep-alive" not in response.content

    asyncio.run(run())


def test_upstream_timeout_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="upstream timeout"):
        create_app(tmp_path, upstream_timeout=0)
    create_app(tmp_path, upstream_timeout=3600.0)
