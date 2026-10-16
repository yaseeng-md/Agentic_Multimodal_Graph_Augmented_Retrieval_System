"""Redis utilities for the AMG Multimodal RAG API.

Features
--------
* Starts Docker Desktop / Docker Engine if needed.
* Creates or starts a persistent Redis container (default host port 6380).
* JSON retrieval and answer cache helpers.
* Fast job-state storage (SQLite / JSON should remain the durable source of truth).
* Atomic fixed-window rate limiting.
* Idempotency reservations for POST /generate and POST /ingest_document.
* Token-checked distributed locks for cross-process coordination.

This file is intentionally self-contained and uses a small RESP2 client built
on Python's standard library, so it does not import redis-py itself. IMPORTANT:
Python may resolve ``import redis`` to this local file instead of the third-party
``redis`` package. If any other dependency needs redis-py, rename this file to
``redis_manager.py`` and use the maintained redis-py client before production.

Configuration (environment variables or .env if python-dotenv is installed)
----------------------------------------------------------------------------
REDIS_HOST=127.0.0.1
REDIS_PORT=6380                
REDIS_DB=0
REDIS_PASSWORD=            
REDIS_CONTAINER_NAME=amg-redis
REDIS_IMAGE=redis:7-alpine
REDIS_VOLUME=amg_redis_data
DOCKER_START_TIMEOUT=20
REDIS_CONNECT_TIMEOUT=2
REDIS_CACHE_TTL=600
REDIS_JOB_TTL=259200             # 3 days
REDIS_IDEMPOTENCY_TTL=86400      # 1 day after job is created
REDIS_IDEMPOTENCY_RESERVATION_TTL=120

Run ``python redis.py`` to start Docker Desktop / the container and verify
Redis connectivity. Importing this module does not start Docker automatically;
call ``ensure_redis_running()`` from FastAPI's lifespan/startup hook.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

# Load a local .env when python-dotenv is present. Environment variables already
# set by the deployment environment take precedence in python-dotenv's defaults.
try:  # pragma: no cover - depends on the deployment environment
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
REDIS_CONTAINER_NAME = os.getenv("REDIS_CONTAINER_NAME", "amg-redis")
REDIS_IMAGE = os.getenv("REDIS_IMAGE", "redis:7-alpine")
REDIS_VOLUME = os.getenv("REDIS_VOLUME", "amg_redis_data")
DOCKER_START_TIMEOUT = int(os.getenv("DOCKER_START_TIMEOUT", "120"))
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "2"))
DEFAULT_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "600"))
DEFAULT_JOB_TTL = int(os.getenv("REDIS_JOB_TTL", "259200"))
DEFAULT_IDEMPOTENCY_TTL = int(os.getenv("REDIS_IDEMPOTENCY_TTL", "86400"))
DEFAULT_IDEMPOTENCY_RESERVATION_TTL = int(
    os.getenv("REDIS_IDEMPOTENCY_RESERVATION_TTL", "120")
)

KEY_PREFIX = "amg"
CACHE_SCHEMA_VERSION = "v1"


class RedisError(RuntimeError):
    """Base exception for Redis connection/protocol/command failures."""


class RedisConnectionError(RedisError):
    """Redis cannot be reached or the connection was interrupted."""


class RedisCommandError(RedisError):
    """Redis returned an error response for a command."""


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode("utf-8")


def _encode_command(parts: Sequence[Any]) -> bytes:
    chunks = [f"*{len(parts)}\r\n".encode("ascii")]
    for part in parts:
        raw = _to_bytes(part)
        chunks.append(f"${len(raw)}\r\n".encode("ascii"))
        chunks.append(raw)
        chunks.append(b"\r\n")
    return b"".join(chunks)


def _read_exactly(stream: Any, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RedisConnectionError("Redis closed the connection unexpectedly.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_response(stream: Any) -> Any:
    line = stream.readline()
    if not line or not line.endswith(b"\r\n"):
        raise RedisConnectionError("Invalid or incomplete response from Redis.")

    prefix, payload = line[:1], line[1:-2]
    if prefix == b"+":
        return payload.decode("utf-8", errors="replace")
    if prefix == b"-":
        raise RedisCommandError(payload.decode("utf-8", errors="replace"))
    if prefix == b":":
        return int(payload)
    if prefix == b"$":
        length = int(payload)
        if length == -1:
            return None
        data = _read_exactly(stream, length)
        if _read_exactly(stream, 2) != b"\r\n":
            raise RedisConnectionError("Invalid bulk-string terminator from Redis.")
        return data.decode("utf-8", errors="replace")
    if prefix == b"*":
        length = int(payload)
        if length == -1:
            return None
        return [_read_response(stream) for _ in range(length)]

    raise RedisConnectionError(f"Unsupported Redis response prefix: {prefix!r}")


class RedisClient:
    """Small synchronous RESP2 client for the commands used in this file.

    Each command uses its own connection, making the client safe to share
    between FastAPI threads without sharing sockets. It is intended for local
    project integration; redis-py is recommended for high-throughput services.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        *,
        password: str | None = REDIS_PASSWORD,
        db: int = REDIS_DB,
        timeout: float = REDIS_CONNECT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.password = password
        self.db = int(db)
        self.timeout = float(timeout)

    def execute_command(self, *parts: Any) -> Any:
        if not parts:
            raise ValueError("A Redis command is required.")
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            ) as sock:
                sock.settimeout(self.timeout)
                with sock.makefile("rb") as stream:
                    if self.password:
                        sock.sendall(_encode_command(("AUTH", self.password)))
                        _read_response(stream)
                    if self.db:
                        sock.sendall(_encode_command(("SELECT", self.db)))
                        _read_response(stream)
                    sock.sendall(_encode_command(parts))
                    return _read_response(stream)
        except RedisError:
            raise
        except (OSError, TimeoutError) as exc:
            raise RedisConnectionError(
                f"Could not connect to Redis at {self.host}:{self.port}: {exc}"
            ) from exc

    def ping(self) -> bool:
        return self.execute_command("PING") == "PONG"

    def get(self, key: str) -> str | None:
        return self.execute_command("GET", key)

    def set(
        self,
        key: str,
        value: Any,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        parts: list[Any] = ["SET", key, value]
        if nx:
            parts.append("NX")
        if ex is not None:
            if int(ex) <= 0:
                raise ValueError("ex must be a positive number of seconds")
            parts.extend(("EX", int(ex)))
        return self.execute_command(*parts) == "OK"

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(self.execute_command("DEL", *keys))

    def exists(self, key: str) -> bool:
        return int(self.execute_command("EXISTS", key)) > 0

    def incr(self, key: str) -> int:
        return int(self.execute_command("INCR", key))

    def expire(self, key: str, seconds: int) -> bool:
        return int(self.execute_command("EXPIRE", key, int(seconds))) == 1

    def ttl(self, key: str) -> int:
        return int(self.execute_command("TTL", key))

    def hset(self, key: str, mapping: Mapping[str, Any]) -> int:
        if not mapping:
            return 0
        parts: list[Any] = ["HSET", key]
        for field, value in mapping.items():
            parts.extend((field, value))
        return int(self.execute_command(*parts))

    def hgetall(self, key: str) -> dict[str, str]:
        values = self.execute_command("HGETALL", key) or []
        return {str(values[i]): str(values[i + 1]) for i in range(0, len(values), 2)}

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        return self.execute_command("EVAL", script, numkeys, *keys_and_args)


# A single client object can be shared because RedisClient opens one connection
# per command. Connections are intentionally not opened until a command runs.
redis_client = RedisClient()


# ---------------------------------------------------------------------------
# Docker Desktop / container lifecycle
# ---------------------------------------------------------------------------


def _run_process(
    args: Sequence[str], *, timeout: int = 15
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RedisError(
            "Docker CLI was not found. Install Docker Desktop and ensure the "
            "'docker' command is available in PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RedisError(f"Command timed out: {' '.join(args)}") from exc


def _docker_daemon_ready() -> bool:
    try:
        result = _run_process(("docker", "info"), timeout=8)
        return result.returncode == 0
    except RedisError:
        return False


def _launch_docker_desktop() -> None:
    """Try to launch Docker Desktop / Docker Engine for common OSes."""
    system = platform.system().lower()
    try:
        if system == "windows":
            roots = [
                os.getenv("ProgramFiles", r"C:\Program Files"),
                os.getenv("ProgramW6432", r"C:\Program Files"),
            ]
            local_app_data = os.getenv("LOCALAPPDATA")
            candidates = [
                Path(root) / "Docker" / "Docker" / "Docker Desktop.exe"
                for root in roots
            ]
            if local_app_data:
                candidates.append(
                    Path(local_app_data)
                    / "Programs"
                    / "Docker"
                    / "Docker"
                    / "Docker Desktop.exe"
                )
            executable = next((p for p in candidates if p.exists()), None)
            if executable is None:
                raise RedisError(
                    "Docker Desktop was not found in the standard Windows paths. "
                    "Open Docker Desktop manually once, or set its install path."
                )
            subprocess.Popen(
                [str(executable)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return

        if system == "darwin":
            subprocess.Popen(
                ["open", "-a", "Docker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return

        # Linux: try a user Docker Desktop service first, then the system engine.
        if shutil.which("systemctl"):
            for command in (
                ("systemctl", "--user", "start", "docker-desktop"),
                ("systemctl", "start", "docker"),
            ):
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=10, check=False
                )
                if result.returncode == 0:
                    return
        raise RedisError(
            "Could not start Docker automatically on Linux. Start Docker Desktop "
            "or Docker Engine manually, then run this script again."
        )
    except RedisError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise RedisError(f"Failed to launch Docker: {exc}") from exc


def _wait_for_docker(timeout_seconds: int = DOCKER_START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "Docker daemon is not ready yet."
    while time.monotonic() < deadline:
        try:
            result = _run_process(("docker", "info"), timeout=8)
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or last_error).strip()
        except RedisError as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RedisError(
        f"Docker did not become ready within {timeout_seconds}s. Last error: "
        f"{last_error}"
    )


def ensure_redis_running(*, timeout_seconds: int = DOCKER_START_TIMEOUT) -> dict[str, Any]:
    """Ensure Docker is running and Redis is reachable; create its container if absent.

    The published host port defaults to 6380 so it does not collide with Qdrant's
    commonly used ports. The container's internal Redis port remains 6379.
    Existing containers are never deleted automatically.
    """
    if not shutil.which("docker"):
        raise RedisError(
            "Docker CLI was not found in PATH. Install Docker Desktop and reopen "
            "your terminal so the 'docker' command becomes available."
        )

    if not _docker_daemon_ready():
        _launch_docker_desktop()
        _wait_for_docker(timeout_seconds)

    inspected = _run_process(
        ("docker", "container", "inspect", "--format", "{{.State.Running}}", REDIS_CONTAINER_NAME),
        timeout=10,
    )

    if inspected.returncode == 0:
        if inspected.stdout.strip().lower() != "true":
            started = _run_process(("docker", "start", REDIS_CONTAINER_NAME), timeout=60)
            if started.returncode != 0:
                raise RedisError(
                    f"Could not start existing container {REDIS_CONTAINER_NAME!r}: "
                    f"{started.stderr.strip()}"
                )
    else:
        command: list[str] = [
            "docker", "run", "-d",
            "--name", REDIS_CONTAINER_NAME,
            "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{REDIS_PORT}:6379",
            "-v", f"{REDIS_VOLUME}:/data",
            REDIS_IMAGE,
            "redis-server", "--appendonly", "yes",
        ]
        if REDIS_PASSWORD:
            command.extend(("--requirepass", REDIS_PASSWORD))
        created = _run_process(command, timeout=300)
        if created.returncode != 0:
            message = (created.stderr or created.stdout).strip()
            raise RedisError(
                f"Could not create Redis container {REDIS_CONTAINER_NAME!r}: {message}. "
                "If a container with this name already exists, inspect it and make "
                "sure its published host port matches REDIS_PORT. This script will "
                "not delete existing containers or data automatically."
            )

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if redis_client.ping():
                return {
                    "status": "ready",
                    "container": REDIS_CONTAINER_NAME,
                    "host": REDIS_HOST,
                    "port": REDIS_PORT,
                    "url": f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
                }
        except RedisError as exc:
            last_error = exc
        time.sleep(1)

    raise RedisError(
        f"Redis container exists but Redis did not respond at "
        f"{REDIS_HOST}:{REDIS_PORT}. Check REDIS_PORT, REDIS_PASSWORD, and Docker "
        f"logs with 'docker logs {REDIS_CONTAINER_NAME}'. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# General JSON cache + stable cache-key helpers
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_cache_key(namespace: str, payload: Any) -> str:
    safe_namespace = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in namespace.lower()
    )
    return f"{KEY_PREFIX}:cache:{CACHE_SCHEMA_VERSION}:{safe_namespace}:{stable_hash(payload)}"


def cache_get_json(key: str) -> Any | None:
    """Return a cached JSON value, or None on a cache miss."""
    raw = redis_client.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Corrupt/old cache entries should not repeatedly break the request.
        redis_client.delete(key)
        raise RedisError(f"Cached JSON is invalid for key {key!r}.") from exc


def cache_set_json(key: str, value: Any, ttl_seconds: int = DEFAULT_CACHE_TTL) -> bool:
    """Store JSON with a TTL. Cache values are disposable and must be recomputable."""
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    return redis_client.set(key, canonical_json(value), ex=ttl_seconds)


def cache_delete(key: str) -> int:
    return redis_client.delete(key)


def retrieval_cache_key(
    *,
    organization: str,
    user_id: str,
    role: str,
    query: str,
    document_ids: Sequence[str] = (),
    version_ids: Sequence[str] = (),
    recursive_search: bool = False,
    top_k: int = 5,
    corpus_revision: str | None = None,
) -> str:
    """Build a tenant/user-scoped retrieval key; pass current version IDs/revision.

    Including the authorization scope prevents accidentally reusing another
    user's results. Prefer identities derived from verified authentication.
    """
    payload = {
        "organization": organization,
        "user_id": user_id,
        "role": role,
        "query": query.strip(),
        "document_ids": sorted(set(document_ids)),
        "version_ids": sorted(set(version_ids)),
        "recursive_search": bool(recursive_search),
        "top_k": int(top_k),
        "corpus_revision": corpus_revision,
    }
    return make_cache_key("retrieval", payload)


def answer_cache_key(
    *,
    organization: str,
    user_id: str,
    role: str,
    query: str,
    retrieval_result: Any,
    model_id: str,
    generation_config: Mapping[str, Any] | None = None,
) -> str:
    """Build an answer key that changes with retrieved evidence/model settings."""
    payload = {
        "organization": organization,
        "user_id": user_id,
        "role": role,
        "query": query.strip(),
        "retrieval_fingerprint": stable_hash(retrieval_result),
        "model_id": model_id,
        "generation_config": dict(generation_config or {}),
    }
    return make_cache_key("answer", payload)


# ---------------------------------------------------------------------------
# Job status cache (SQLite/JSON remain durable sources of truth)
# ---------------------------------------------------------------------------


def _job_key(job_type: str, job_id: str) -> str:
    safe_type = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_type)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
    return f"{KEY_PREFIX}:job:{safe_type}:{safe_id}"


def set_job_state(
    job_type: str,
    job_id: str,
    state: Mapping[str, Any],
    ttl_seconds: int = DEFAULT_JOB_TTL,
) -> bool:
    return cache_set_json(_job_key(job_type, job_id), dict(state), ttl_seconds)


def get_job_state(job_type: str, job_id: str) -> dict[str, Any] | None:
    state = cache_get_json(_job_key(job_type, job_id))
    return state if isinstance(state, dict) else None


def delete_job_state(job_type: str, job_id: str) -> int:
    return redis_client.delete(_job_key(job_type, job_id))


# ---------------------------------------------------------------------------
# Atomic fixed-window rate limiting
# ---------------------------------------------------------------------------

_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {count, ttl}
"""


def check_rate_limit(
    identity: str,
    *,
    bucket: str,
    limit: int,
    window_seconds: int = 60,
) -> dict[str, Any]:
    """Atomically count a request and return allowed/count/remaining/retry_after."""
    if limit < 1 or window_seconds < 1:
        raise ValueError("limit and window_seconds must be positive")
    # Hash identity so raw IPs/user identifiers are not visible in Redis key names.
    identity_digest = stable_hash(identity)[:32]
    safe_bucket = "".join(c if c.isalnum() or c in "-_" else "_" for c in bucket)
    key = f"{KEY_PREFIX}:ratelimit:{CACHE_SCHEMA_VERSION}:{safe_bucket}:{identity_digest}"
    result = redis_client.eval(_RATE_LIMIT_SCRIPT, 1, key, window_seconds)
    count, ttl = int(result[0]), max(int(result[1]), 1)
    return {
        "allowed": count <= limit,
        "limit": limit,
        "count": count,
        "remaining": max(limit - count, 0),
        "retry_after": ttl,
        "key": key,
    }


def get_rate_limit_identity(request: Any) -> str:
    """Use a verified identity if auth middleware set it; otherwise use client IP.

    Set request.state.authenticated_user_id only after validating a JWT. Do not
    trust user_id from an unverified JSON body for authorization/rate limiting.
    """
    state = getattr(request, "state", None)
    authenticated_user_id = getattr(state, "authenticated_user_id", None)
    if authenticated_user_id:
        return f"user:{authenticated_user_id}"
    client = getattr(request, "client", None)
    client_host = getattr(client, "host", None) if client else None
    if client_host:
        return f"ip:{client_host}"
    return "ip:unknown"


def enforce_rate_limit(
    request: Any,
    response: Any,
    *,
    bucket: str,
    limit: int,
    window_seconds: int = 60,
) -> dict[str, Any]:
    """FastAPI route helper. Raises HTTP 429 when over limit, 503 if Redis fails."""
    # Import inside the function so the utility module can be used standalone.
    from fastapi import HTTPException

    try:
        decision = check_rate_limit(
            get_rate_limit_identity(request),
            bucket=bucket,
            limit=limit,
            window_seconds=window_seconds,
        )
    except RedisError as exc:
        # Fail closed so Redis outages do not silently remove the generation guard.
        raise HTTPException(
            status_code=503,
            detail="Rate-limiting service is temporarily unavailable.",
        ) from exc

    headers = {
        "X-RateLimit-Limit": str(decision["limit"]),
        "X-RateLimit-Remaining": str(decision["remaining"]),
        "X-RateLimit-Reset-After": str(decision["retry_after"]),
    }
    if not decision["allowed"]:
        headers["Retry-After"] = str(decision["retry_after"])
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry later.",
            headers=headers,
        )
    for name, value in headers.items():
        response.headers[name] = value
    return decision


# ---------------------------------------------------------------------------
# Idempotency for POST /generate and POST /ingest_document
# ---------------------------------------------------------------------------

_RESERVE_IDEMPOTENCY_SCRIPT = """
local existing = redis.call('GET', KEYS[1])
if existing then
    return {0, existing}
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
return {1, ARGV[1]}
"""

_UPDATE_IDEMPOTENCY_SCRIPT = """
local existing = redis.call('GET', KEYS[1])
if not existing then
    return 0
end
local record = cjson.decode(existing)
if tostring(record['job_id']) ~= ARGV[1] then
    return -1
end
record['status'] = ARGV[2]
if ARGV[3] ~= '' then
    record['error'] = ARGV[3]
else
    record['error'] = nil
end
redis.call('SET', KEYS[1], cjson.encode(record), 'EX', ARGV[4])
return 1
"""


def _idempotency_key(identity: str, endpoint: str, client_key: str) -> str:
    identity_digest = stable_hash(identity)[:32]
    client_key_digest = hashlib.sha256(client_key.encode("utf-8")).hexdigest()
    safe_endpoint = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in endpoint.lower()
    )
    return (
        f"{KEY_PREFIX}:idempotency:{CACHE_SCHEMA_VERSION}:"
        f"{identity_digest}:{safe_endpoint}:{client_key_digest}"
    )


def reserve_idempotency(
    *,
    identity: str,
    endpoint: str,
    idempotency_key: str,
    payload: Any,
    job_id: str,
    reservation_ttl_seconds: int = DEFAULT_IDEMPOTENCY_RESERVATION_TTL,
) -> dict[str, Any]:
    """Atomically reserve a logical request.

    Returns state='new' when this caller owns the reservation, 'existing' for a
    retry with the same payload, and 'conflict' when a key is reused with a
    different payload. The caller should create the durable job only for 'new'.

    Once the SQLite job is safely created, call update_idempotency_status(...)
    with status='queued' and the longer idempotency TTL. The short reservation
    TTL prevents a crash before durable job creation from blocking retries forever.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise ValueError("idempotency_key must not be empty")
    if len(idempotency_key) > 255:
        raise ValueError("idempotency_key must be at most 255 characters")
    if reservation_ttl_seconds < 1:
        raise ValueError("reservation_ttl_seconds must be positive")

    key = _idempotency_key(identity, endpoint, idempotency_key)
    request_hash = stable_hash(payload)
    record = {
        "request_hash": request_hash,
        "job_id": job_id,
        "status": "reserved",
        "endpoint": endpoint,
        "created_at": int(time.time()),
    }
    result = redis_client.eval(
        _RESERVE_IDEMPOTENCY_SCRIPT,
        1,
        key,
        canonical_json(record),
        reservation_ttl_seconds,
    )
    created = int(result[0]) == 1
    try:
        existing_record = json.loads(str(result[1]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RedisError("Invalid idempotency record stored in Redis.") from exc

    if created:
        state = "new"
    elif existing_record.get("request_hash") != request_hash:
        state = "conflict"
    else:
        state = "existing"
    return {"state": state, "record": existing_record, "redis_key": key}


def update_idempotency_status(
    *,
    identity: str,
    endpoint: str,
    idempotency_key: str,
    job_id: str,
    status: str,
    ttl_seconds: int = DEFAULT_IDEMPOTENCY_TTL,
    error: str | None = None,
) -> bool:
    """Update a reserved key only if it still belongs to this job ID."""
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be positive")
    key = _idempotency_key(identity, endpoint, idempotency_key)
    result = redis_client.eval(
        _UPDATE_IDEMPOTENCY_SCRIPT,
        1,
        key,
        job_id,
        status,
        error or "",
        ttl_seconds,
    )
    if int(result) == -1:
        raise RedisError("Idempotency reservation belongs to a different job.")
    return int(result) == 1


# ---------------------------------------------------------------------------
# Distributed locks (optional for multi-process GPU / shared work)
# ---------------------------------------------------------------------------

_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

_RENEW_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""


def acquire_lock(name: str, ttl_seconds: int = 120, token: str | None = None) -> str | None:
    """Acquire a distributed lock, returning its unique ownership token or None."""
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be positive")
    lock_key = f"{KEY_PREFIX}:lock:{name}"
    token = token or uuid.uuid4().hex
    return token if redis_client.set(lock_key, token, nx=True, ex=ttl_seconds) else None


def release_lock(name: str, token: str) -> bool:
    """Release only if this caller still owns the lock."""
    lock_key = f"{KEY_PREFIX}:lock:{name}"
    return int(redis_client.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, token)) == 1


def renew_lock(name: str, token: str, ttl_seconds: int = 120) -> bool:
    """Extend an owned lock; never renew another worker's lock."""
    if ttl_seconds < 1:
        raise ValueError("ttl_seconds must be positive")
    lock_key = f"{KEY_PREFIX}:lock:{name}"
    return int(
        redis_client.eval(_RENEW_LOCK_SCRIPT, 1, lock_key, token, ttl_seconds)
    ) == 1


def health_check() -> dict[str, Any]:
    """Return Redis connectivity information without starting Docker."""
    try:
        is_ready = bool(redis_client.ping())
    except (RedisError, OSError):
        is_ready = False
    return {
        "status": "ready" if is_ready else "unavailable",
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
    }


def main() -> int:
    try:
        info = ensure_redis_running()
        print("Redis is ready:")
        print(json.dumps(info, indent=2))
        print("Test command: docker exec -it " + REDIS_CONTAINER_NAME + " redis-cli ping")
        return 0
    except RedisError as exc:
        print(f"Redis startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
