#!/bin/bash

NUM_CLIENTS=${1:-5}

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENT_INDEX=$(printf "%02d" "$i")
    CLIENT_ID="client_${CLIENT_INDEX}"
    CONTAINER_NAME="task_afed_client_${CLIENT_INDEX}"

    echo "Starting ${CLIENT_ID}..."

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --network task_afed_default \
        --gpus all \
        -e CLIENT_ID="${CLIENT_ID}" \
        -e CLIENT_BROKER_HOST="${CLIENT_HOST}" \
        -e CLIENT_BROKER_PORT="5674" \
        -e SERVER_HOST="task_afed_server" \
        -e SERVER_PORT="8080" \
        -e TASK_ID="afed" \
        -e LOCAL_TEST="1" \
        -e BASE_WEIGHTS_PATH="/data/cityscapes/yolo11n.pt" \
        -e DATA_YAML_PATH="/data/cityscapes/data.yaml" \
        -e LOCAL_CKPT_PATH="/tmp/yolo_local_client.pt" \
        -e BATCH_SIZE="8" \
        -e NUM_CLASSES="1" \
        -v "${DATA_PATH}:/data" \
        task_afed_client_image
done
