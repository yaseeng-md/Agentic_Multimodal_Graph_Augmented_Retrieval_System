
import os
import subprocess
import time
from pathlib import Path

import docker
import httpx
from dotenv import load_dotenv
from docker.errors import NotFound

# Load .env from this file's directory, regardless of where
# the Python process was launched.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CONTAINER_NAME = os.getenv("QDRANT_CONTAINER_NAME", "amg-qdrant")
IMAGE = os.getenv("QDRANT_IMAGE", "qdrant/qdrant:latest")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
STORAGE_VOLUME = os.getenv(
    "QDRANT_STORAGE_VOLUME", "amg_qdrant_storage"
)

HTTP_HOST = os.getenv("QDRANT_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
GRPC_HOST = os.getenv("QDRANT_GRPC_HOST", "127.0.0.1")
GRPC_PORT = int(os.getenv("QDRANT_GRPC_HOST_PORT", "6334"))

DOCKER_TIMEOUT = int(os.getenv("QDRANT_DOCKER_START_TIMEOUT", "120"))
API_TIMEOUT = int(os.getenv("QDRANT_API_READY_TIMEOUT", "60"))

# Only stop Qdrant on shutdown if this process started it.
_started_by_this_process = False


def start_docker_desktop():
    """Connect to Docker, starting Docker Desktop if necessary."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except docker.errors.DockerException:
        pass

    if os.getenv("QDRANT_START_DOCKER_DESKTOP", "true").lower() == "true":
        print("Starting Docker Desktop...")
        subprocess.run(
            ["systemctl", "--user", "start", "docker-desktop"],
            check=True,
        )
    else:
        raise RuntimeError("Docker is unavailable and auto-start is disabled.")

    for _ in range(DOCKER_TIMEOUT // 2):
        try:
            client = docker.from_env()
            client.ping()
            print("Docker is ready.")
            return client
        except docker.errors.DockerException:
            time.sleep(2)

    raise RuntimeError("Docker daemon did not become ready.")


def wait_for_qdrant():
    """Wait until the Qdrant REST API responds."""
    print(f"Waiting for Qdrant at {QDRANT_URL}...")

    deadline = time.monotonic() + API_TIMEOUT

    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                f"{QDRANT_URL.rstrip('/')}/collections",
                headers=(
                    {"api-key": QDRANT_API_KEY}
                    if QDRANT_API_KEY else None
                ),
                timeout=2,
            )
            response.raise_for_status()
            print("Qdrant is ready.")
            return
        except httpx.HTTPError:
            time.sleep(2)

    raise RuntimeError(
        f"Qdrant did not respond at {QDRANT_URL} "
        f"within {API_TIMEOUT} seconds."
    )


def start_qdrant():
    """Create or start Qdrant, then wait for its API."""
    global _started_by_this_process

    client = start_docker_desktop()

    try:
        client.images.get(IMAGE)
    except NotFound:
        print(f"Pulling image: {IMAGE}")
        client.images.pull(IMAGE)

    try:
        container = client.containers.get(CONTAINER_NAME)
        container.reload()

        print(f"Found container: {CONTAINER_NAME}")

        if container.status != "running":
            container.start()
            _started_by_this_process = True

    except NotFound:
        print("Creating Qdrant container...")

        # Create the persistent volume explicitly.
        client.volumes.create(name=STORAGE_VOLUME)

        environment = {}
        if QDRANT_API_KEY:
            environment["QDRANT__SERVICE__API_KEY"] = QDRANT_API_KEY

        container = client.containers.run(
            image=IMAGE,
            name=CONTAINER_NAME,
            detach=True,
            ports={
                "6333/tcp": (HTTP_HOST, HTTP_PORT),
                "6334/tcp": (GRPC_HOST, GRPC_PORT),
            },
            volumes={
                STORAGE_VOLUME: {
                    "bind": "/qdrant/storage",
                    "mode": "rw",
                }
            },
            environment=environment or None,
        )
        _started_by_this_process = True

    try:
        wait_for_qdrant()
        return container
    except Exception:
        # Don't leave a newly started container running after
        # this startup attempt fails.
        if _started_by_this_process:
            container.stop(timeout=10)
            _started_by_this_process = False
        raise


def stop_qdrant():
    """Stop only the container this process started."""
    global _started_by_this_process

    if not _started_by_this_process:
        print("Qdrant was already running independently; leaving it up.")
        return

    try:
        client = docker.from_env()
        container = client.containers.get(CONTAINER_NAME)
        container.reload()

        if container.status == "running":
            print("Stopping Qdrant...")
            container.stop(timeout=10)

    except NotFound:
        print("Qdrant container does not exist.")
    finally:
        _started_by_this_process = False


if __name__ == "__main__":
    # start_qdrant()
    stop_qdrant()