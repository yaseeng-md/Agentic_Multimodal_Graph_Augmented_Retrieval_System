"""Neo4j Docker container lifecycle management."""

import logging
import os
import time
from typing import Any, Optional

import docker
from docker.errors import NotFound, APIError
from neo4j import GraphDatabase

from graph.config import (
    NEO4J_CONTAINER_NAME,
    NEO4J_IMAGE,
    NEO4J_VOLUME,
    NEO4J_HTTP_PORT,
    NEO4J_BOLT_PORT,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_DATABASE,
)

logger = logging.getLogger("amg-neo4j-manager")
_started_by_this_process = False


def _get_docker_client() -> docker.DockerClient:
    return docker.from_env()


def is_neo4j_ready() -> bool:
    """Quick non-blocking check whether Neo4j is responding to queries."""
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            connection_timeout=2.0,
        )
        with driver.session(database=NEO4J_DATABASE) as session:
            res = session.run("RETURN 1 AS ok").single()
            if res and res["ok"] == 1:
                driver.close()
                return True
        driver.close()
        return False
    except Exception:
        return False


def wait_for_neo4j(timeout_seconds: int = 60, interval_seconds: float = 2.0) -> None:
    """Poll Neo4j Bolt port until ready or timeout expires."""
    start_time = time.time()
    last_err: Optional[Exception] = None

    while time.time() - start_time < timeout_seconds:
        if is_neo4j_ready():
            logger.info("Neo4j Bolt connection is ready at %s", NEO4J_URI)
            return
        time.sleep(interval_seconds)

    raise TimeoutError(f"Neo4j readiness check timed out after {timeout_seconds}s. Last error: {last_err}")


def ensure_neo4j_running(timeout_seconds: int = 60) -> dict[str, Any]:
    """Ensure Neo4j container is running and ready for Bolt connections."""
    global _started_by_this_process

    if is_neo4j_ready():
        logger.info("Neo4j Bolt connection is already ready at %s", NEO4J_URI)
        return {
            "container": NEO4J_CONTAINER_NAME,
            "bolt_uri": NEO4J_URI,
            "status": "ready",
        }

    if os.getenv("RUNNING_IN_DOCKER", "").lower() in ("true", "1", "yes"):
        wait_for_neo4j(timeout_seconds=timeout_seconds)
        return {
            "container": NEO4J_CONTAINER_NAME,
            "bolt_uri": NEO4J_URI,
            "status": "ready",
        }

    client = _get_docker_client()

    try:
        container = client.containers.get(NEO4J_CONTAINER_NAME)
        if container.status != "running":
            logger.info("Starting existing Neo4j container '%s'...", NEO4J_CONTAINER_NAME)
            container.start()
            _started_by_this_process = True
        else:
            logger.debug("Neo4j container '%s' is already running.", NEO4J_CONTAINER_NAME)
    except NotFound:
        logger.info("Creating and starting new Neo4j container '%s'...", NEO4J_CONTAINER_NAME)
        try:
            client.images.get(NEO4J_IMAGE)
        except NotFound:
            logger.info("Pulling Neo4j image '%s'...", NEO4J_IMAGE)
            client.images.pull(NEO4J_IMAGE)

        container = client.containers.run(
            NEO4J_IMAGE,
            name=NEO4J_CONTAINER_NAME,
            detach=True,
            ports={
                "7474/tcp": ("127.0.0.1", NEO4J_HTTP_PORT),
                "7687/tcp": ("127.0.0.1", NEO4J_BOLT_PORT),
            },
            volumes={
                NEO4J_VOLUME: {"bind": "/data", "mode": "rw"},
            },
            environment={
                "NEO4J_AUTH": f"{NEO4J_USERNAME}/{NEO4J_PASSWORD}",
                "NEO4J_PLUGINS": '["apoc"]',
            },
        )
        _started_by_this_process = True

    wait_for_neo4j(timeout_seconds=timeout_seconds)
    return {
        "container": NEO4J_CONTAINER_NAME,
        "bolt_uri": NEO4J_URI,
        "http_port": NEO4J_HTTP_PORT,
        "status": "ready",
    }


def stop_neo4j() -> None:
    """Stop the Neo4j container if it was started by this process."""
    global _started_by_this_process
    if not _started_by_this_process:
        return

    client = _get_docker_client()
    try:
        container = client.containers.get(NEO4J_CONTAINER_NAME)
        if container.status == "running":
            logger.info("Stopping Neo4j container '%s'...", NEO4J_CONTAINER_NAME)
            container.stop()
            _started_by_this_process = False
    except NotFound:
        pass
    except APIError as exc:
        logger.warning("Error stopping Neo4j container: %s", exc)
