try:
    import os
    from time import sleep

    from flwr.client import NumPyClient, start_client
    from ftdl.ftdl import FTDL
    from grpc import _channel as ch
    from task import (DEVICE, Net, get_weights, load_data, probe_loss,
                      set_weights, test, train)
except Exception as e:
    print("Could not import task module:", e)
    ftdl = FTDL()
    ftdl.announce_error(e)
    print("Waiting for termination...")
    while(True): 
        pass 

class FlowerClient(NumPyClient):

    def get_properties(self, config):
        return {
            "client_id": int(client_id),
            "num_examples": int(len(trainloader.dataset)),
            "pool_size": int(pool_size),
            "dirichlet_alpha": float(dirichlet_alpha),
        }


    def fit(self, parameters, config):
        global net

        net = set_weights(net, parameters)

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

        results["client_id"] = int(client_id)
        results["probe_loss"] = float(probe_value)

        return (
            get_weights(net),
            len(trainloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        global ftdl, last_acc, net
        net = set_weights(net, parameters)
        accuracy = test(net, testloader)
        ftdl.send_stats(new_round=True, acc_contrib=accuracy-last_acc)
        last_acc = accuracy
        return accuracy, len(testloader.dataset), {"metric": accuracy}

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

        ftdl = FTDL()

        net = Net().to(DEVICE)

        client_id = int(
    os.environ["CLIENT_ID"]
        )

        pool_size = int(
            os.environ.get("POOL_SIZE", "50")
        )

        dirichlet_alpha = float(
            os.environ.get("DIRICHLET_ALPHA", "0.3")
        )

        partition_seed = int(
            os.environ.get("PARTITION_SEED", "2048")
        )

        print(
            f"Iniciando cliente {client_id}: "
            f"pool_size={pool_size}, "
            f"alpha={dirichlet_alpha}, "
            f"seed={partition_seed}",
            flush=True,
        )

        trainloader, probe_loader, testloader = load_data(
            data_path="/data",
            client_id=client_id,
            n_clients=pool_size,
            alpha=dirichlet_alpha,
            seed=partition_seed,
            )
        
        main_client()
        
        ftdl.announce_finish()
    
    except Exception as e:
        print(e)
        ftdl.announce_error(e) 

    print("Waiting for termination...")
    while(True): 
        pass # Waiting termination