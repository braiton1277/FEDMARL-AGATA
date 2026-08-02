try:
    import os
    import random
    from contextlib import contextmanager
    from pathlib import Path
    from time import sleep

    import numpy as np
    import torch
    from flwr.client import NumPyClient, start_client
    from ftdl.ftdl import FTDL
    from grpc import _channel as ch
    from task import (DEVICE, Net, get_weights, load_data, probe_loss,
                      set_label_attack, set_weights, test, train)

    try:
        import fcntl
    except ImportError:
        fcntl = None

except Exception as e:
    print("Could not import task module:", e)
    ftdl = FTDL()
    ftdl.announce_error(e)
    print("Waiting for termination...")
    while True:
        pass


CPU_DEVICE = torch.device("cpu")


def select_attacker_ids(
    n_clients: int,
    attack_fraction: float,
    seed: int,
) -> set[int]:
    """
    Sorteia os atacantes de forma determinística, como no código original.
    Todos os clientes executam esta mesma função e obtêm a mesma lista.
    """
    if n_clients <= 0:
        raise ValueError(
            "n_clients deve ser maior que zero."
        )

    if not 0.0 <= attack_fraction <= 1.0:
        raise ValueError(
            "ATTACK_FRACTION deve estar entre 0 e 1."
        )

    n_attackers = int(
        round(attack_fraction * n_clients)
    )

    if n_attackers == 0:
        return set()

    rng = np.random.RandomState(
        int(seed) + 999
    )

    selected = rng.choice(
        np.arange(n_clients),
        size=n_attackers,
        replace=False,
    )

    return {
        int(client_id)
        for client_id in selected.tolist()
    }


@contextmanager
def acquire_compute_slot():
    """
    Limita quantos clientes podem executar probe/treino na GPU ao mesmo tempo.

    Em CPU, nenhum bloqueio é aplicado. Em CUDA, os contêineres disputam
    arquivos de lock compartilhados pelo volume /gpu_slots.
    """
    if DEVICE.type != "cuda":
        yield None
        return

    if fcntl is None:
        raise RuntimeError(
            "fcntl não está disponível. O controle de vagas de GPU "
            "requer um contêiner Linux."
        )

    lock_directory = Path(gpu_lock_dir)
    lock_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    acquired_file = None
    acquired_slot = None

    print(
        f"[Cliente {client_id}] aguardando uma das "
        f"{max_concurrent_gpu_clients} vagas de GPU...",
        flush=True,
    )

    try:
        while acquired_file is None:
            first_slot = (
                client_id % max_concurrent_gpu_clients
            )

            slot_order = [
                (first_slot + offset)
                % max_concurrent_gpu_clients
                for offset in range(
                    max_concurrent_gpu_clients
                )
            ]

            for slot in slot_order:
                lock_path = (
                    lock_directory
                    / f"gpu_slot_{slot}.lock"
                )

                candidate_file = open(
                    lock_path,
                    mode="a+",
                    encoding="utf-8",
                )

                try:
                    fcntl.flock(
                        candidate_file.fileno(),
                        fcntl.LOCK_EX
                        | fcntl.LOCK_NB,
                    )

                    candidate_file.seek(0)
                    candidate_file.truncate()
                    candidate_file.write(
                        f"client_id={client_id}\n"
                        f"pid={os.getpid()}\n"
                    )
                    candidate_file.flush()

                    acquired_file = candidate_file
                    acquired_slot = slot
                    break

                except BlockingIOError:
                    candidate_file.close()

            if acquired_file is None:
                sleep(
                    0.20
                    + random.random() * 0.10
                )

        print(
            f"[Cliente {client_id}] adquiriu "
            f"GPU_SLOT={acquired_slot}",
            flush=True,
        )

        yield acquired_slot

    finally:
        if acquired_file is not None:
            try:
                acquired_file.seek(0)
                acquired_file.truncate()
                acquired_file.flush()
            finally:
                fcntl.flock(
                    acquired_file.fileno(),
                    fcntl.LOCK_UN,
                )
                acquired_file.close()

            print(
                f"[Cliente {client_id}] liberou "
                f"GPU_SLOT={acquired_slot}",
                flush=True,
            )


def move_model_back_to_cpu():
    """
    Retira o modelo da GPU antes de liberar a vaga compartilhada.
    """
    global net

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    net = net.to(CPU_DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


class FlowerClient(NumPyClient):

    def get_properties(self, config):
        return {
            "client_id": int(client_id),
            "num_examples": int(
                len(trainloader.dataset)
            ),
            "pool_size": int(pool_size),
            "dirichlet_alpha": float(
                dirichlet_alpha
            ),
            "is_attacker": int(is_attacker),
            "attack_start_round": int(
                attack_start_round
            ),
            "configured_flip_rate": float(
                configured_flip_rate
            ),
            "attack_fraction": float(
                attack_fraction
            ),
            "attacker_count": int(
                len(attacker_ids)
            ),
            "attack_seed": int(
                partition_seed + 999
            ),
            "device": str(DEVICE),
            "max_concurrent_gpu_clients": int(
                max_concurrent_gpu_clients
            ),
        }

    def fit(self, parameters, config):
        global net

        server_round = int(
            config.get("server_round", 0)
        )

        attack_active = bool(
            is_attacker
            and server_round
            >= attack_start_round
        )

        applied_flip_rate = (
            configured_flip_rate
            if attack_active
            else 0.0
        )

        attack_status = set_label_attack(
            trainloader,
            enabled=attack_active,
            rate=applied_flip_rate,
        )

        if (
            probe_loader.dataset
            is not trainloader.dataset
        ):
            set_label_attack(
                probe_loader,
                enabled=attack_active,
                rate=applied_flip_rate,
            )

        role = (
            "ATACANTE"
            if is_attacker
            else "HONESTO"
        )

        print(
            f"[Cliente {client_id}] "
            f"rodada={server_round}, "
            f"papel={role}, "
            f"ataque_ativo={int(attack_active)}, "
            f"flip_rate={applied_flip_rate:.3f}, "
            f"rotulos_invertidos="
            f"{attack_status['active_flips']}/"
            f"{attack_status['num_examples']}",
            flush=True,
        )

        net = set_weights(
            net.to(CPU_DEVICE),
            parameters,
        )

        with acquire_compute_slot():
            try:
                net = net.to(DEVICE)

                probe_value = probe_loss(
                    net=net,
                    probe_loader=probe_loader,
                    max_batches=5,
                    device=DEVICE,
                )

                results = train(
                    net,
                    trainloader,
                    steps=10,
                    lr=0.01,
                    device=DEVICE,
                )

            finally:
                move_model_back_to_cpu()

        results["client_id"] = int(
            client_id
        )
        results["probe_loss"] = float(
            probe_value
        )
        results["server_round"] = int(
            server_round
        )
        results["is_attacker"] = int(
            is_attacker
        )
        results["attack_active"] = int(
            attack_active
        )
        results["flip_rate"] = float(
            applied_flip_rate
        )
        results["flipped_examples"] = int(
            attack_status["active_flips"]
        )

        return (
            get_weights(net),
            len(trainloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        global ftdl, last_acc, net

        net = set_weights(
            net.to(CPU_DEVICE),
            parameters,
        )

        with acquire_compute_slot():
            try:
                net = net.to(DEVICE)

                accuracy = test(
                    net,
                    testloader,
                    device=DEVICE,
                )

            finally:
                move_model_back_to_cpu()

        ftdl.send_stats(
            new_round=True,
            acc_contrib=accuracy - last_acc,
        )

        last_acc = accuracy

        return (
            accuracy,
            len(testloader.dataset),
            {"metric": accuracy},
        )


def main_client():
    for _ in range(60):
        try:
            start_client(
                server_address=(
                    f"{os.environ['SERVER_HOST']}:"
                    f"{os.environ['SERVER_PORT']}"
                ),
                client=FlowerClient().to_client(),
            )
            return

        except ch._MultiThreadedRendezvous as e:
            print(e)
            print(
                "Retrying in 30 seconds..."
            )
            sleep(30)


if __name__ == "__main__":

    try:
        last_acc = 0

        ftdl = FTDL()

        client_id = int(
            os.environ["CLIENT_ID"]
        )

        pool_size = int(
            os.environ.get(
                "POOL_SIZE",
                "50",
            )
        )

        dirichlet_alpha = float(
            os.environ.get(
                "DIRICHLET_ALPHA",
                "0.3",
            )
        )

        partition_seed = int(
            os.environ.get(
                "PARTITION_SEED",
                "2048",
            )
        )

        attack_fraction = float(
            os.environ.get(
                "ATTACK_FRACTION",
                "0.2",
            )
        )

        attacker_ids = select_attacker_ids(
            n_clients=pool_size,
            attack_fraction=attack_fraction,
            seed=partition_seed,
        )

        attack_start_round = int(
            os.environ.get(
                "ATTACK_START_ROUND",
                "1",
            )
        )

        configured_flip_rate = float(
            os.environ.get(
                "FLIP_RATE",
                "1.0",
            )
        )

        max_concurrent_gpu_clients = int(
            os.environ.get(
                "MAX_CONCURRENT_GPU_CLIENTS",
                "2",
            )
        )

        gpu_lock_dir = os.environ.get(
            "GPU_LOCK_DIR",
            "/gpu_slots",
        )

        if not 0.0 <= attack_fraction <= 1.0:
            raise ValueError(
                "ATTACK_FRACTION deve estar entre 0 e 1."
            )

        if attack_start_round <= 0:
            raise ValueError(
                "ATTACK_START_ROUND deve ser maior "
                "que zero."
            )

        if not 0.0 <= configured_flip_rate <= 1.0:
            raise ValueError(
                "FLIP_RATE deve estar entre 0 e 1."
            )

        if max_concurrent_gpu_clients <= 0:
            raise ValueError(
                "MAX_CONCURRENT_GPU_CLIENTS deve ser "
                "maior que zero."
            )

        is_attacker = (
            client_id in attacker_ids
        )

        # O modelo nasce em CPU e só vai para CUDA após adquirir uma vaga.
        net = Net().to(CPU_DEVICE)

        print(
            f"Iniciando cliente {client_id}: "
            f"pool_size={pool_size}, "
            f"alpha={dirichlet_alpha}, "
            f"seed={partition_seed}, "
            f"papel="
            f"{'ATACANTE' if is_attacker else 'HONESTO'}, "
            f"attack_fraction="
            f"{attack_fraction}, "
            f"attackers_total="
            f"{len(attacker_ids)}, "
            f"attack_start_round="
            f"{attack_start_round}, "
            f"flip_rate="
            f"{configured_flip_rate}, "
            f"device={DEVICE}, "
            f"max_gpu_clients="
            f"{max_concurrent_gpu_clients}",
            flush=True,
        )

        if client_id == 0:
            print(
                f"Atacantes sorteados automaticamente: "
                f"{sorted(attacker_ids)}",
                flush=True,
            )

        trainloader, probe_loader, testloader = (
            load_data(
                data_path="/data",
                client_id=client_id,
                n_clients=pool_size,
                alpha=dirichlet_alpha,
                seed=partition_seed,
            )
        )

        main_client()

        ftdl.announce_finish()

    except Exception as e:
        print(e)
        ftdl.announce_error(e)

    print("Waiting for termination...")

    while True:
        pass