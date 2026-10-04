from fit_supervisionado import executar_fit
from grpc import RpcError
import traceback
import cache_isolado
import leitura_retry
try:
    import multiprocessing as mp
    import os
    import gc
    import torch
    from time import sleep

    from flwr.client import NumPyClient, start_client
    from ftdl.ftdl import FTDL
    from grpc import _channel as ch
    from task import (DEVICE, Net, get_weights, load_data, load_local_probe_loader,
    probe_loss, set_weights, test, train)



except Exception as e:
    print("Could not import task module:", e)
    ftdl = FTDL()
    ftdl.announce_error(e)
    print("Waiting for termination...")
    while True:
        pass


LOCAL_TEST = os.environ.get("LOCAL_TEST", "0") == "1"


class LocalFTDL:
    def __init__(self, side="client"):
        self.task_id = os.environ.get("TASK_ID", "local_yolo")
        print(f"[LOCAL_TEST] FTDL desativado ({side})")

    def send_stats(self, *args, **kwargs):
        pass

    def announce_finish(self, *args, **kwargs):
        print("[LOCAL_TEST] Cliente finalizado")

    def announce_error(self, exception):
        print(f"[LOCAL_TEST] Erro no cliente: {exception}")




def _fit_worker(parameters, conn):
    try:
        worker_net = Net().to(DEVICE)
        worker_net = set_weights(worker_net, parameters)

        worker_probe_loader = load_local_probe_loader(
            "/data",
            batch_size=2,
        )

        probe_value = probe_loss(
            net=worker_net,
            probe_loader=worker_probe_loader,
            max_batches=5,
            device=DEVICE,
        )

        worker_trainloader, _ = load_data("/data")

        results = train(
            worker_net,
            None,
            None,
            epochs=1,
            device=DEVICE,
        )

        output_weights = get_weights(worker_net)
        num_examples = len(worker_trainloader.dataset)

        conn.send(
            (
                "ok",
                output_weights,
                num_examples,
                results,
                float(probe_value),
            )
        )

    except Exception as e:
        traceback.print_exc()
        conn.send(("error", repr(e)))

    finally:
        conn.close()





class FlowerClient(NumPyClient):
    def fit(self, parameters, config):
        message = executar_fit(parameters)

        if message[0] != "ok":
            raise RuntimeError(
                f"YOLO fit worker falhou: {message[1]}"
            )

        _, output_weights, num_examples, results, probe_value = message

        client_id = os.environ.get(
            "CLIENT_ID",
            os.environ.get("HOSTNAME", "unknown"),
        )

        results["client_id"] = client_id
        results["probe_loss"] = float(probe_value)

        print(
            f"[CLIENT ID] {client_id}",
            flush=True,
        )

        print(
            f"[FIT WORKER EXIT] {client_id}: subprocesso encerrado",
            flush=True,
        )

        return output_weights, num_examples, results



    def evaluate(self, parameters, config):
        global ftdl, last_acc, net

        net = set_weights(net, parameters)

        map50 = test(net, testloader)
        eval_loss = 1.0 - map50

        ftdl.send_stats(
            new_round=True,
            acc_contrib=map50 - last_acc
            )

        last_acc = map50

        return (
            float(eval_loss),
            len(testloader.dataset),
            {"map50": float(map50)}
            )


def main_client():
    for tries in range(60):
        try:
            start_client(
                server_address=f"{os.environ['SERVER_HOST']}:{os.environ['SERVER_PORT']}",
                client=FlowerClient().to_client(),
            )
            return
        except (RpcError, RuntimeError, OSError, EOFError) as exc:
            print(
                f"[CLIENT RECONNECT] tentativa={tries + 1}/60 "
                f"erro={exc!r}",
                flush=True,
            )
            traceback.print_exc()
            sleep(10)
    raise RuntimeError("Limite de tentativas de reconexão atingido")


if __name__ == "__main__":
    try:
        last_acc = 0

        ftdl = LocalFTDL() if LOCAL_TEST else FTDL()

        net = Net().to(DEVICE)

        trainloader, testloader = load_data("/data")

        probe_loader = load_local_probe_loader(
            "/data",
            batch_size=2,
        )

        print(
            f"[PROBE LOCAL] loader carregado, "
            f"batches={len(probe_loader)}",
            flush=True,
            )



        main_client()

        ftdl.announce_finish()

    except Exception as e:
        print(e)
        ftdl.announce_error(e)


if not LOCAL_TEST and __name__ == "__main__":
    print("Waiting for termination...")
    while True:
        sleep(1)
