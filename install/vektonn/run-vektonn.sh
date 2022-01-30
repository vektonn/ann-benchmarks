#!/bin/bash
set -e

export VEKTONN_INDEX_NAME=ann-benchmark.index
export VEKTONN_INDEX_VERSION=1.0
export VEKTONN_INDEX_SHARD_ID=SingleShard

THIS_SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
DOCKER_COMPOSE_FILE=$THIS_SCRIPT_DIR/docker-compose.yaml

docker-compose --file "$DOCKER_COMPOSE_FILE" down || true # swallow network removal error
docker-compose --file "$DOCKER_COMPOSE_FILE" up --detach
