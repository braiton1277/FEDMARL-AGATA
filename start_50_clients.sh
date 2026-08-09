#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PROJECT_NAME="${PROJECT_NAME:-task_fedmarl}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"

export CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
export DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data}"
export GPU_LOCK_PATH="${GPU_LOCK_PATH:-${ROOT_DIR}/runtime/gpu_slots}"

export POOL_SIZE="${POOL_SIZE:-${NUM_CLIENTS}}"
export DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
export PARTITION_SEED="${PARTITION_SEED:-2048}"

export ATTACK_START_ROUND="${ATTACK_START_ROUND:-1}"
export FLIP_RATE="${FLIP_RATE:-1.0}"
export MAX_CONCURRENT_GPU_CLIENTS="${MAX_CONCURRENT_GPU_CLIENTS:-2}"

if [[ -z "${ATTACK_FRACTION:-}" ]]; then
    echo "ERRO: defina ATTACK_FRACTION antes de iniciar."
    echo
    echo "Para 20% de atacantes:"
    echo 'export ATTACK_FRACTION=0.2'
    echo
    echo "Para 60% de atacantes:"
    echo 'export ATTACK_FRACTION=0.6'
    exit 1
fi

docker exec -i task_fedmarl-task_cebd_server-1 python - "${ATTACK_FRACTION}" <<'PY'
import sys

value = float(sys.argv[1])

if not 0.0 <= value <= 1.0:
    raise SystemExit(
        "ERRO: ATTACK_FRACTION deve estar entre 0 e 1."
    )
PY

ATTACKER_IDS_PREVIEW="$(
docker exec -i task_fedmarl-task_cebd_server-1 python - "${NUM_CLIENTS}" "${ATTACK_FRACTION}" "${PARTITION_SEED}" <<'PY'
import sys
import numpy as np

n_clients = int(sys.argv[1])
fraction = float(sys.argv[2])
seed = int(sys.argv[3])

n_attackers = int(round(fraction * n_clients))
rng = np.random.RandomState(seed + 999)

if n_attackers == 0:
    selected = []
else:
    selected = sorted(
        int(x)
        for x in rng.choice(
            np.arange(n_clients),
            size=n_attackers,
            replace=False,
        ).tolist()
    )

print(",".join(map(str, selected)))
PY
)"

if ! [[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: NUM_CLIENTS deve ser inteiro positivo."
    exit 1
fi

if ! [[ "${MAX_CONCURRENT_GPU_CLIENTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERRO: MAX_CONCURRENT_GPU_CLIENTS deve ser inteiro positivo."
    exit 1
fi

mkdir -p \
    "${DATA_PATH}" \
    "${GPU_LOCK_PATH}"

echo "=============================================="
echo "Iniciando clientes FEDMARL"
echo "NUM_CLIENTS=${NUM_CLIENTS}"
echo "POOL_SIZE=${POOL_SIZE}"
echo "ATTACK_FRACTION=${ATTACK_FRACTION}"
echo "ATTACKERS_AUTOMATICOS=${ATTACKER_IDS_PREVIEW}"
echo "ATTACK_START_ROUND=${ATTACK_START_ROUND}"
echo "FLIP_RATE=${FLIP_RATE}"
echo "MAX_CONCURRENT_GPU_CLIENTS=${MAX_CONCURRENT_GPU_CLIENTS}"
echo "DATA_PATH=${DATA_PATH}"
echo "GPU_LOCK_PATH=${GPU_LOCK_PATH}"
echo "=============================================="

echo
echo "Construindo a imagem dos clientes..."
docker compose \
    -p "${PROJECT_NAME}" \
    -f client/compose.yaml \
    build task_cebd_client

echo
echo "Removendo contêineres antigos dos clientes..."
for client_id in $(seq 0 $((NUM_CLIENTS - 1))); do
    docker rm -f \
        "${PROJECT_NAME}-client-${client_id}" \
        >/dev/null 2>&1 || true
done

start_client() {
    local client_id="$1"
    local container_name="${PROJECT_NAME}-client-${client_id}"

    docker compose \
        -p "${PROJECT_NAME}" \
        -f client/compose.yaml \
        run -d \
        --name "${container_name}" \
        -e CLIENT_ID="${client_id}" \
        -e POOL_SIZE="${POOL_SIZE}" \
        -e DIRICHLET_ALPHA="${DIRICHLET_ALPHA}" \
        -e PARTITION_SEED="${PARTITION_SEED}" \
        -e ATTACK_FRACTION="${ATTACK_FRACTION}" \
        -e ATTACK_START_ROUND="${ATTACK_START_ROUND}" \
        -e FLIP_RATE="${FLIP_RATE}" \
        -e MAX_CONCURRENT_GPU_CLIENTS="${MAX_CONCURRENT_GPU_CLIENTS}" \
        task_cebd_client \
        >/dev/null

    echo "Cliente ${client_id} iniciado: ${container_name}"
}

echo
echo "Iniciando primeiro o cliente 0 para preparar o CIFAR-10..."
start_client 0

echo "Aguardando o cliente 0 carregar os dados..."
deadline=$((SECONDS + 600))

while true; do
    if docker logs "${PROJECT_NAME}-client-0" 2>&1 \
        | grep -q "\[Cliente 0\] amostras="; then
        echo "CIFAR-10 e partição do cliente 0 prontos."
        break
    fi

    if ! docker ps \
        --format '{{.Names}}' \
        | grep -qx "${PROJECT_NAME}-client-0"; then
        echo "ERRO: o cliente 0 encerrou antes de carregar os dados."
        docker logs "${PROJECT_NAME}-client-0" || true
        exit 1
    fi

    if (( SECONDS >= deadline )); then
        echo "ERRO: tempo esgotado aguardando o cliente 0."
        docker logs "${PROJECT_NAME}-client-0" || true
        exit 1
    fi

    sleep 2
done

echo
echo "Iniciando os demais clientes..."
for client_id in $(seq 1 $((NUM_CLIENTS - 1))); do
    start_client "${client_id}"
done

echo
echo "Todos os ${NUM_CLIENTS} clientes foram iniciados."
echo
echo "Acompanhe o servidor com:"
echo "docker logs -f ${PROJECT_NAME}-task_cebd_server-1"
echo
echo "Acompanhe a GPU com:"
echo "watch -n 1 nvidia-smi"