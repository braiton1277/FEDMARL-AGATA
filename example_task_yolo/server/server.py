try:
    import os
    from typing import List, Tuple

    from flwr.common import (Metrics, ndarrays_to_parameters,
                             parameters_to_ndarrays)
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg
    from ftdl.ftdl import FTDL
    from task import *
except Exception as e:
    print("Could not import module:", e)
    ftdl = FTDL(side="server")
    ftdl.announce_error(e)
    print("Waiting for termination...")
    while(True): 
        pass 




LOCAL_TEST = os.environ.get("LOCAL_TEST", "0") == "1"


class LocalFTDL:
    def __init__(self, side="server"):
        self.task_id = os.environ.get("TASK_ID", "local_yolo")
        print(f"[LOCAL_TEST] FTDL desativado ({side})")

    def announce_finish(self, *args, **kwargs):
        print("[LOCAL_TEST] Servidor finalizado")

    def announce_error(self, exception):
        print(f"[LOCAL_TEST] Erro no servidor: {exception}")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global task_reporter, comm_round
    examples = [num_examples for num_examples, _ in metrics]

    examples = [num_examples for num_examples, _ in metrics]
    losses = [num_examples * m["metric"] for num_examples, m in metrics]
    result = {"metric": sum(losses) / sum(examples)}
    
    print(f"metric = {result}, round = {comm_round}")
    return result

class ModFedAvg(FedAvg):
    def aggregate_fit(self, server_rounds, results, failures):
        global net, comm_round
        comm_round = server_rounds
        print(f"Aggregate fit, round {server_rounds}, results {results}, failures {failures}")
        aggregated_parameters, metrics = super().aggregate_fit(server_rounds, results, failures)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            net = set_weights(net, ndarrays)
        
        try:
            with open(f"/model/{ftdl.task_id}.pickle", "wb") as f:
                f.write(net_to_bytes(net))
        except Exception as e:
            print(f"{e}: Could not save model")

        return aggregated_parameters, metrics

def get_strategy():
    global net

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    return ModFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    return ModFedAvg(
        fraction_fit=1.0,  # Select all available clients
        fraction_evaluate=1.0,  # Disable evaluation
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

if __name__ == "__main__":

    try:
        ftdl = LocalFTDL(side="server") if LOCAL_TEST else FTDL(side="server")

        
        # Fix after
        num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
        num_rounds = int(os.environ.get("NUM_ROUNDS", "5"))

        config = ServerConfig(num_rounds=num_rounds)

        net = Net()

        try:
            with open(f"/model/{ftdl.task_id}.pickle", "rb") as f:
                net = bytes_to_net(net, f.read())
        except Exception as e:
            print(f"{e}: Could not load model from file. Starting from scratch.")

        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=get_strategy(),
        )

        ftdl.announce_finish(net_to_bytes(net))

    except Exception as e:
        print(e)
        ftdl.announce_error(e)
    
    print("Waiting for termination...")
    while(True): 
        pass 

