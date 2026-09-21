import docker 

docker_client = docker.client.from_env()

qdrant_contianer = docker_client.containers.get("4057d454491a62e42fd30d48ae575ead910c95702013d270ce71cc8b9bd86a7e")
qdrant_contianer.start()
