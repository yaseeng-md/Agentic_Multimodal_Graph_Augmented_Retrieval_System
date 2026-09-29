
import subprocess
import time

import docker
import httpx
from docker.errors import NotFound

CONTAINER_NAME = "amg-qdrant"
IMAGE = "qdrant/qdrant:latest"

QDRANT_URL = "http://localhost:6333"
QDRANT_GRPC_PORT = 6334
STORAGE_VOLUME = "amg_qdrant_storage"


def start_docker_desktop():
    """Start Docker Desktop if the daemon isn't available."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except docker.errors.DockerException:
        pass

    print("Starting Docker Desktop...")

    subprocess.run(
        ["systemctl", "--user", "start", "docker-desktop"],
        check=True,
    )

    # Wait for Docker Desktop to become ready.
    for _ in range(60):
        try:
            client = docker.from_env()
            client.ping()
            print("Docker is ready.")
            return client
        except docker.errors.DockerException:
            time.sleep(2)

    raise RuntimeError("Docker daemon did not become ready.")


def start_qdrant():
    """Create or start the Qdrant container and wait for its API."""
    client = start_docker_desktop()

    # Pull the image only if it isn't already available locally.
    try:
        client.images.get(IMAGE)
    except NotFound:
        print(f"Pulling {IMAGE}...")
        client.images.pull(IMAGE)

    try:
        container = client.containers.get(CONTAINER_NAME)
        print(f"Found existing container: {CONTAINER_NAME}")

        if container.status != "running":
            container.start()

    except NotFound:
        print("Creating Qdrant container...")

        container = client.containers.run(
            image=IMAGE,
            name=CONTAINER_NAME,
            detach=True,
            ports={
                "6333/tcp": ("127.0.0.1", 6333),
                "6334/tcp": ("127.0.0.1", QDRANT_GRPC_PORT),
            },
            volumes={
                STORAGE_VOLUME: {
                    "bind": "/qdrant/storage",
                    "mode": "rw",
                }
            },
        )

    # Wait for the REST API to respond.
    print("Waiting for Qdrant...")

    for _ in range(30):
        try:
            response = httpx.get(
                f"{QDRANT_URL}/collections",
                timeout=2,
            )
            response.raise_for_status()
            print("Qdrant is ready.")
            return container
        except (httpx.HTTPError, httpx.TimeoutException):
            time.sleep(2)

    raise RuntimeError("Qdrant did not become ready.")


def stop_qdrant():
    """Stop the managed Qdrant container without deleting its data."""
    try:
        client = docker.from_env()
        container = client.containers.get(CONTAINER_NAME)

        if container.status == "running":
            print("Stopping Qdrant...")
            container.stop(timeout=10)

    except NotFound:
        print("Qdrant container does not exist.")



if __name__ == "__main__":
    start_qdrant()