#!/usr/bin/env python3
"""Public agent tool with one stable request ID per logical environment action."""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROTOCOL = "alfworld-step-request-id-v1"


class RequestProtocolError(RuntimeError):
    pass


class AmbiguousResponseError(OSError):
    pass


class ActionResponseError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"Environment returned HTTP {status}: {json.dumps(body, ensure_ascii=False)}")
        self.status, self.body = status, body


def http_json(method, url, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if payload is not None and "request_id" in payload:
        headers["X-Request-ID"] = payload["request_id"]
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    except http.client.HTTPException as error:
        raise AmbiguousResponseError("Environment connection ended before a valid response") from error
    with response:
        try:
            raw = response.read()
        except http.client.HTTPException as error:
            raise AmbiguousResponseError("Environment response was incomplete") from error
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            if 200 <= response.status < 300:
                raise AmbiguousResponseError("Environment returned invalid JSON") from None
            body = {"error": raw.decode(errors="replace")[:1000]}
        return response.status, body


def checked_base(base):
    parts = urllib.parse.urlsplit(base or "")
    if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1")
            or parts.username or parts.password or parts.query or parts.fragment
            or not re.fullmatch(r"/session/[0-9a-f]{32}", parts.path)):
        raise RequestProtocolError("A loopback ALFWorld session URL is required")
    try:
        port = parts.port
    except ValueError:
        raise RequestProtocolError("An explicit valid loopback port is required") from None
    if port is None:
        raise RequestProtocolError("An explicit loopback port is required")
    return base.rstrip("/"), urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def request_action(base, action, *, transport=http_json, request_id=None, timeout=15.0,
                   total_timeout=900.0, poll_interval=0.2, clock=time.monotonic, sleep=time.sleep, trace=None):
    base, health_url = checked_base(base)
    action = action.strip()
    if not action:
        raise ValueError("An action is required")
    request_id = request_id or uuid.uuid4().hex
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("Invalid request ID")
    if (not all(math.isfinite(value) for value in (timeout, total_timeout, poll_interval))
            or timeout <= 0 or total_timeout <= 0 or poll_interval < 0):
        raise ValueError("Retry bounds must be positive")
    trace = trace if trace is not None else {}
    trace.update(request_id=request_id, posts=0, lookups=0, health_checks=0)
    started, deadline = clock(), clock() + total_timeout

    def remaining():
        left = deadline - clock()
        if left <= 0:
            raise TimeoutError("Environment outcome remains unknown for request " + request_id)
        return min(timeout, left)

    def pause():
        sleep(min(poll_interval, remaining()))

    def health(expected=None):
        while True:
            trace["health_checks"] += 1
            try:
                status, body = transport("GET", health_url, None, remaining())
                break
            except (OSError, TimeoutError, urllib.error.URLError):
                pause()
        if (status != 200 or not isinstance(body, dict) or body.get("ready") is not True
                or body.get("action_request_protocol") != PROTOCOL
                or not isinstance(body.get("server_instance_id"), str) or not body["server_instance_id"]):
            raise RequestProtocolError("Environment does not advertise the request-ID protocol")
        if expected is not None and body["server_instance_id"] != expected:
            raise RequestProtocolError("Environment server changed while the action outcome was unknown")
        return body["server_instance_id"]

    try:
        instance = health()
        mode = "post"
        while True:
            if mode == "post":
                trace["posts"] += 1
                try:
                    status, body = transport("POST", base + "/step",
                                             {"action": action, "request_id": request_id}, remaining())
                except (OSError, TimeoutError, urllib.error.URLError):
                    mode = "lookup"
                    continue
                if not 200 <= status < 300:
                    raise ActionResponseError(status, body)
                return body
            trace["lookups"] += 1
            try:
                status, body = transport("GET", base + "/requests/" + request_id, None, remaining())
            except (OSError, TimeoutError, urllib.error.URLError):
                pause()
                continue
            if status == 404:
                health(instance)
                mode = "post"  # Only a confirmed unknown ID permits same-ID resubmission.
                continue
            if status == 202:
                if (not isinstance(body, dict) or body.get("request_id") != request_id
                        or body.get("status") != "pending"):
                    raise RequestProtocolError("Invalid pending request envelope")
                pause()
                continue
            if status == 200:
                if (not isinstance(body, dict) or body.get("request_id") != request_id
                        or body.get("status") != "completed" or type(body.get("response_status")) is not int
                        or not 100 <= body["response_status"] <= 599 or "result" not in body):
                    raise RequestProtocolError("Invalid completed request envelope")
                if not 200 <= body["response_status"] < 300:
                    raise ActionResponseError(body["response_status"], body["result"])
                return body["result"]
            if status < 500:
                raise ActionResponseError(status, body)
            pause()  # A lookup failure never authorizes action execution.
    finally:
        trace["elapsed_seconds"] = clock() - started


def main(argv=None, environ=None):
    argv, environ = (sys.argv[1:] if argv is None else argv), (os.environ if environ is None else environ)
    trace = {}
    try:
        base, _ = checked_base(environ.get("ALFWORLD_SESSION_URL"))
        action = " ".join(argv).strip()
        if action in ("", "--observe"):
            status, state = http_json("GET", base + "/observe", None, 60)
            if not 200 <= status < 300:
                raise ActionResponseError(status, state)
        else:
            state = request_action(base, action, trace=trace)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except (OSError, TimeoutError, RequestProtocolError, ActionResponseError) as error:
        print(str(error), file=sys.stderr)
        return 75
    finally:
        if trace and environ.get("ALFWORLD_REQUEST_TRACE") == "1":
            print("ALFWORLD_REQUEST_TIMING " + json.dumps(trace, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
