try:
    import os
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


class FlowerClient(NumPyClient):
    def fit(self, parameters, config):
        global net
        net = set_weights(net, parameters)

        probe_value = probe_loss(
            net=net,
            probe_loader=probe_loader,
            max_batches=5,
            device=DEVICE,
            )

        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        results["client_id"] = os.environ.get("HOSTNAME", "unknown")

        results["probe_loss"] = float(
            probe_value
            )


        return get_weights(net), len(trainloader.dataset), results


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
        except ch._MultiThreadedRendezvous as e:
            print(e)
            print("Retrying in 30 seconds...")
            sleep(30)


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


if not LOCAL_TEST:
    print("Waiting for termination...")
    while True:
        sleep(1)
