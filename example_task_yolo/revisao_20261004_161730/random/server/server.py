try:
    import os
    import torch.nn.functional as F
    from typing import List, Tuple
    import random
    import numpy as np
    from flwr.common import (Metrics, ndarrays_to_parameters,
                             parameters_to_ndarrays)
    from flwr.server import Server,  ServerConfig, start_server
    from flwr.server.client_manager import SimpleClientManager
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




class PrioritizedReplayJoint:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        d_in: int,
        alpha: float = 0.6,
        eps: float = 1e-3,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)

        self.alpha = float(alpha)
        self.eps = float(eps)

        self.obs = np.zeros(
            (self.capacity, self.n_agents, self.d_in),
            dtype=np.float32,
        )

        self.act = np.zeros(
            (self.capacity, self.n_agents),
            dtype=np.uint8,
        )

        self.reward = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.obs2 = np.zeros(
            (self.capacity, self.n_agents, self.d_in),
            dtype=np.float32,
        )

        self.done = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.priorities = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.n = 0
        self.max_priority = 1.0

        self.rng = np.random.default_rng(
            int(seed)
        )

    def add(
        self,
        obs,
        act,
        reward,
        obs2,
        done: bool,
    ):
        obs = np.asarray(obs, dtype=np.float32)
        act = np.asarray(act, dtype=np.uint8)
        obs2 = np.asarray(obs2, dtype=np.float32)

        expected_obs_shape = (
            self.n_agents,
            self.d_in,
        )

        expected_action_shape = (
            self.n_agents,
        )

        if obs.shape != expected_obs_shape:
            raise ValueError(
                f"obs possui dimensão {obs.shape}, "
                f"esperado {expected_obs_shape}."
            )

        if obs2.shape != expected_obs_shape:
            raise ValueError(
                f"obs2 possui dimensão {obs2.shape}, "
                f"esperado {expected_obs_shape}."
            )

        if act.shape != expected_action_shape:
            raise ValueError(
                f"act possui dimensão {act.shape}, "
                f"esperado {expected_action_shape}."
            )

        if not np.isfinite(float(reward)):
            raise ValueError(
                "A recompensa não é finita."
            )

        index = self.position

        self.obs[index] = obs
        self.act[index] = act
        self.reward[index] = float(reward)
        self.obs2[index] = obs2
        self.done[index] = (
            1.0 if done else 0.0
        )

        self.priorities[index] = (
            self.max_priority
        )

        self.position = (
            self.position + 1
        ) % self.capacity

        self.n = min(
            self.n + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ):

        if self.n == 0:
            raise RuntimeError(
                "O replay buffer está vazio."
            )

        batch_size = min(
            int(batch_size),
            self.n,
        )

        priorities = (
            self.priorities[:self.n]
            .astype(np.float64)
        )

        probabilities = (
            priorities + self.eps
        ) ** self.alpha

        probability_sum = float(
            probabilities.sum()
        )

        if probability_sum <= 0.0:
            probabilities = (
                np.ones_like(probabilities)
                / len(probabilities)
            )
        else:
            probabilities /= probability_sum

        indices = self.rng.choice(
            self.n,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        weights = (
            self.n * probabilities[indices]
        ) ** (-float(beta))

        weights /= (
            weights.max() + 1e-12
        )

        return (
            self.obs[indices],
            self.act[indices],
            self.reward[indices],
            self.obs2[indices],
            self.done[indices],
            indices.astype(np.int64),
            weights.astype(np.float32),
        )



    def update_priorities(
        self,
        indices,
        td_error_absolute,
    ):
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        td_error_absolute = np.asarray(
            td_error_absolute,
            dtype=np.float32,
        )

        if indices.shape != td_error_absolute.shape:
            raise ValueError(
                "indices e td_error_absolute "
                "possuem dimensões incompatíveis."
            )

        new_priorities = (
            td_error_absolute + self.eps
        )

        self.priorities[
            indices
        ] = new_priorities

        self.max_priority = float(
            max(
                self.max_priority,
                float(
                    new_priorities.max(
                        initial=0.0
                    )
                ),
            )
        )


    def state_dict(self):
        return {
            "obs": self.obs[:self.n].copy(),
            "act": self.act[:self.n].copy(),
            "reward": self.reward[:self.n].copy(),
            "obs2": self.obs2[:self.n].copy(),
            "done": self.done[:self.n].copy(),
            "priorities": self.priorities[:self.n].copy(),

            "position": self.position,
            "n": self.n,
            "max_priority": self.max_priority,

            "rng_state": self.rng.bit_generator.state,
        }




    def load_state_dict(self, state):
        n = int(state["n"])

        if n > self.capacity:
            raise ValueError(
                f"Replay salvo possui {n} transições, "
                f"mas capacity={self.capacity}."
            )

        self.obs[:n] = state["obs"]
        self.act[:n] = state["act"]
        self.reward[:n] = state["reward"]
        self.obs2[:n] = state["obs2"]
        self.done[:n] = state["done"]
        self.priorities[:n] = state["priorities"]

        self.n = n
        self.position = int(state["position"])
        self.max_priority = float(state["max_priority"])

        self.rng.bit_generator.state = state["rng_state"]



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


reference_momentum = None
MOMENTUM_BETA = 0.9





VDN_GAMMA = 0.90
VDN_GRAD_CLIP = 1.0
VDN_TARGET_SYNC_EVERY = 2






VDN_WARMUP_TRANSITIONS = 50
VDN_SELECTION_SEED = 4096


completed_rounds = 0

client_staleness = {}
client_streak = {}
validation_loss_history = []
REWARD_WINDOW = 5

def update_staleness_streak(
    staleness,
    streak,
    all_client_ids,
    selected_ids,
):
    selected_set = set(selected_ids)



    for client_id in all_client_ids:
        if client_id in selected_set:
            staleness[client_id] = 0.0
            streak[client_id] = streak.get(client_id, 0) + 1
        else:
            staleness[client_id] = staleness.get(client_id, 0.0) + 1.0
            streak[client_id] = 0



def get_parameter_indices(net):
    state_keys = list(
        net.model.model.state_dict().keys()
    )

    parameter_keys = set(
        dict(net.model.model.named_parameters()).keys()
    )

    return [
        i
        for i, key in enumerate(state_keys)
        if key in parameter_keys
    ]





def flatten_ndarrays(ndarrays):
    return np.concatenate(
        [
            np.asarray(array, dtype=np.float32).reshape(-1)
            for array in ndarrays
        ]
    )



def build_context_matrix_vdn(
    client_ids,
    client_projections,
    client_probe_losses,
    staleness,
    streak,
):
    projection = np.array(
        [
            client_projections.get(client_id, 0.0)
            for client_id in client_ids
        ],
        dtype=np.float32,
    )

    probe = np.array(
        [
            client_probe_losses.get(client_id, 0.0)
            for client_id in client_ids
        ],
        dtype=np.float32,
    )

    stale = np.array(
        [
            staleness.get(client_id, 0.0)
            for client_id in client_ids
        ],
        dtype=np.float32,
    )

    streak_values = np.array(
        [
            streak.get(client_id, 0)
            for client_id in client_ids
        ],
        dtype=np.float32,
    )

    stale_max = float(
        stale.max() + 1e-6
    )

    staleness_normalized = (
        stale / stale_max
    ).astype(np.float32)

    streak_cap = 5.0

    streak_normalized = np.clip(
        streak_values / streak_cap,
        0.0,
        1.0,
    ).astype(np.float32)

    bias = np.array(
        [float(cid in client_probe_losses) for cid in client_ids],
        dtype=np.float32,
    )

    state_matrix = np.stack(
        [
            bias,
            projection,
            probe,
            staleness_normalized,
            streak_normalized,
        ],
        axis=1,
    ).astype(np.float32)

    return state_matrix



class AgentMLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)




class VDNSelector:
    def __init__(
        self,
        gamma=0.90,
        grad_clip=1.0,
        target_sync_every=2,
        warmup_transitions=2,
        selection_seed=4096,
    ):
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.target_sync_every = target_sync_every
        self.warmup_transitions = warmup_transitions
        self.selection_seed = selection_seed

        self.q = None
        self.q_tgt = None
        self.optimizer = None

        self.replay_buffer = None
        self.pending_transition = None

        self.train_calls = 0



    def state_dict(self):
        return {
            "q": (
                self.q.state_dict()
                if self.q is not None
                else None
            ),
            "q_tgt": (
                self.q_tgt.state_dict()
                if self.q_tgt is not None
                else None
            ),
            "optimizer": (
                self.optimizer.state_dict()
                if self.optimizer is not None
                else None
            ),
            "replay_buffer": (
                self.replay_buffer.state_dict()
                if self.replay_buffer is not None
                else None
            ),
            "train_calls": self.train_calls,
            "pending_transition": self.pending_transition,
        }

    def load_state_dict(self, state):
        rng_state = torch.random.get_rng_state()

        torch.manual_seed(self.selection_seed)

        self.q = AgentMLP(
            d_in=5,
            hidden=128,
        ).to(DEVICE)

        self.q_tgt = AgentMLP(
            d_in=5,
            hidden=128,
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(
            self.q.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        torch.random.set_rng_state(rng_state)

        if state["q"] is not None:
            self.q.load_state_dict(state["q"])

        if state["q_tgt"] is not None:
            self.q_tgt.load_state_dict(state["q_tgt"])

        self.q_tgt.eval()

        if state["optimizer"] is not None:
            self.optimizer.load_state_dict(
                state["optimizer"]
            )

        replay_state = state.get("replay_buffer")

        if replay_state is not None:
            n_agents = int(
                replay_state["obs"].shape[1]
            )

            d_in = int(
                replay_state["obs"].shape[2]
            )

            self.replay_buffer = PrioritizedReplayJoint(
                capacity=20000,
                n_agents=n_agents,
                d_in=d_in,
                alpha=0.6,
                eps=1e-3,
                seed=self.selection_seed,
            )

            self.replay_buffer.load_state_dict(
                replay_state
            )

        else:
            self.replay_buffer = None

        self.train_calls = int(
            state.get("train_calls", 0)
        )

        self.pending_transition = state.get(
            "pending_transition"
        )



vdn_selector = VDNSelector()


def experiment_state_dict(
    logical_round,
    client_ids,
):
    return {
        "completed_rounds": int(logical_round),

        "client_ids": list(client_ids),

        "reference_momentum": (
            reference_momentum.copy()
            if reference_momentum is not None
            else None
        ),

        "client_staleness": dict(
            client_staleness
        ),

        "client_streak": dict(
            client_streak
        ),

        "validation_loss_history": list(
            validation_loss_history
        ),

        "vdn": vdn_selector.state_dict(),
    }


def windowed_reward(
    loss_history,
    new_loss: float,
    window_size: int = 5,
):
    if len(loss_history) == 0:
        base_loss = float(new_loss)
    else:
        window = min(
            window_size,
            len(loss_history),
        )

        base_loss = float(
            np.mean(
                loss_history[-window:]
            )
        )

    reward = (
        base_loss - float(new_loss)
    ) / (
        abs(base_loss) + 1e-6
    )

    if not np.isfinite(reward):
        raise RuntimeError(
            "A recompensa não é finita."
        )

    return float(reward)




VDN_PER_BETA_START = 0.4
VDN_PER_BETA_END = 1.0
VDN_PER_BETA_STEPS = 4000





class ModFedAvg(FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        if not getattr(self, "_initial_pool_ready", False):
            expected = int(os.environ["NUM_CLIENTS"])
            print(f"[WAIT CLIENTS] aguardando {expected} clientes", flush=True)
            if not client_manager.wait_for(expected, timeout=1800):
                raise RuntimeError(
                    f"Não conectaram os {expected} clientes em 30 minutos."
                )
            self._initial_pool_ready = True
        return super().configure_fit(
            server_round, parameters, client_manager
        )

    def aggregate_fit(self, server_rounds, results, failures):
        global net, comm_round, reference_momentum, completed_rounds
        global validation_loss_history
        logical_round = completed_rounds + server_rounds
        comm_round = logical_round



        if failures:
            print(f"[FIT FAILURES] count={len(failures)}")
            for i, failure in enumerate(failures):
                print(
                    f"[FIT FAILURE {i}] "
                    f"type={type(failure).__name__}, "
                    f"repr={failure!r}"
                )




        selection_mode = os.environ.get(
            "SELECTION_MODE",
            "random",
        ).lower()


        available_client_ids = sorted([
            fit_res.metrics.get("client_id", "unknown")
            for _, fit_res in results])

        all_client_ids = [
            f"client_{i:02d}"
            for i in range(int(os.environ["NUM_CLIENTS"]))
        ]
        available_set = set(available_client_ids)
        expected_set = set(all_client_ids)

        if len(available_set) != len(available_client_ids):
            raise RuntimeError("Resposta duplicada de cliente.")

        unexpected = available_set - expected_set
        if unexpected:
            raise RuntimeError(f"IDs inesperados: {sorted(unexpected)}")

        minimum = int(os.environ["K_SELECTED"])
        if len(available_set) < minimum:
            raise RuntimeError(
                f"Apenas {len(available_set)} respostas válidas; "
                f"são necessárias {minimum}. Checkpoint anterior preservado."
            )

        missing = sorted(expected_set - available_set)
        print(
            f"[AVAILABILITY] received={len(available_set)} "
            f"expected={len(all_client_ids)} missing={missing}",
            flush=True,
        )


        if len(set(all_client_ids)) != len(all_client_ids):
            raise RuntimeError(
                f"client_ids duplicados: {all_client_ids}"
                )

        print(
            f"[CLIENT ORDER] {all_client_ids}"
            )


        global_ndarrays = get_weights(net)

        parameter_indices = get_parameter_indices(net)

        global_vector = flatten_ndarrays(
            [
                global_ndarrays[i]
                for i in parameter_indices
            ]
        )


        client_deltas = {}
        client_probe_losses = {}

        for _, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")

            probe_value = fit_res.metrics.get(
            "probe_loss"
            )

            if probe_value is None:
                raise RuntimeError(
                f"Cliente {client_id} não retornou probe_loss."
                 )

            probe_value = float(probe_value)

            if not np.isfinite(probe_value):
                raise RuntimeError(
                    f"probe_loss inválido para {client_id}: "
                    f"{probe_value}"
                )

            client_probe_losses[client_id] = probe_value





            candidate_ndarrays = parameters_to_ndarrays(
            fit_res.parameters
            )


            candidate_vector = flatten_ndarrays(
                [
                    candidate_ndarrays[i]
                    for i in parameter_indices
                ]
            )

            client_delta = candidate_vector - global_vector
            client_deltas[client_id] = client_delta

            delta_norm = float(
                np.linalg.norm(client_delta)
                )

            print(
                f"[DELTA] client={client_id}, "
                f"norm={delta_norm:.6f}"
                )





        for client_id, probe_value in client_probe_losses.items():
            print(
                f"[PROBE LOSS] client={client_id}, "
                f"loss={probe_value:.6f}"
            )


        reference_gradient = server_reference_grad(
            net=net,
            probe_loader=probe_loader,
            batches=1,
            )

        print(
            f"[PROBE] reference_gradient="
            f"{reference_gradient.size}"
            )

        for client_id, client_delta in client_deltas.items():

            print(
                f"[PROBE] client={client_id}, "
                f"delta={client_delta.size}, "
                f"gradient={reference_gradient.size}"
                )

            if client_delta.shape != reference_gradient.shape:
                raise RuntimeError(
                    f"Dimensões incompatíveis para {client_id}: "
                    f"delta={client_delta.shape}, "
                    f"gradient={reference_gradient.shape}"
        )


        if reference_momentum is None:
            reference_momentum = reference_gradient.copy()
        else:
            reference_momentum = (
                MOMENTUM_BETA * reference_momentum
                + (1.0 - MOMENTUM_BETA) * reference_gradient
                )

        momentum_norm = float(
            np.linalg.norm(reference_momentum)
            )

        if momentum_norm <= 0.0:
            raise RuntimeError(
                "O momento do gradiente é nulo."
                 )

        descent_momentum_direction = (
            -reference_momentum / momentum_norm
            )

        client_projections = {}

        for client_id, client_delta in client_deltas.items():

            projection_mom = float(
                np.dot(
                    client_delta,
                    descent_momentum_direction,
                )
            )

            client_projections[client_id] = projection_mom

            print(
                f"[PROJECTION] client={client_id}, "
                f"projection_mom={projection_mom:+.6f}"
            )


        k_selected = int(
            os.environ.get("K_SELECTED", "2")
        )

        if selection_mode == "vdn":
            state_matrix = build_context_matrix_vdn(
                client_ids=all_client_ids,
                client_projections=client_projections,
                client_probe_losses=client_probe_losses,
                staleness=client_staleness,
                streak=client_streak,
                )




            if vdn_selector.replay_buffer is None:
                vdn_selector.replay_buffer = PrioritizedReplayJoint(
                    capacity=20000,
                    n_agents=len(all_client_ids),
                    d_in=5,
                    alpha=0.6,
                    eps=1e-3,
                    seed=4096,
                )

                print(
                    f"[REPLAY] criado: "
                    f"capacity={vdn_selector.replay_buffer.capacity}, "
                    f"n_agents={vdn_selector.replay_buffer.n_agents}, "
                    f"d_in={vdn_selector.replay_buffer.d_in}"
                )




            for index, client_id in enumerate(all_client_ids):
                print(
                    f"[VDN STATE] client={client_id}, "
                    f"state={state_matrix[index].tolist()}"
                    )




            if vdn_selector.q is None:
                rng_state = torch.random.get_rng_state()
                torch.manual_seed(4096)

                vdn_selector.q = AgentMLP(
                    d_in=5,
                    hidden=128,
                ).to(DEVICE)

                vdn_selector.q_tgt = AgentMLP(
                    d_in=5,
                    hidden=128,
                ).to(DEVICE)

                vdn_selector.q_tgt.load_state_dict(
                    vdn_selector.q.state_dict()
                )

                vdn_selector.q_tgt.eval()

                vdn_selector.optimizer = torch.optim.Adam(
                    vdn_selector.q.parameters(),
                    lr=1e-3,
                    weight_decay=1e-4,
                )

                torch.random.set_rng_state(
                    rng_state
                )

                print("[VDN INIT] q + q_tgt + optimizer criados")


            vdn_selector.q.eval()

            with torch.no_grad():
                state_tensor = torch.tensor(
                    state_matrix,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                q_values = vdn_selector.q(state_tensor)

            q_values_np = (
                q_values
                .detach()
                .cpu()
                .numpy()
            )

            for index, client_id in enumerate(all_client_ids):
                q0 = float(q_values_np[index, 0])
                q1 = float(q_values_np[index, 1])
                advantage = q1 - q0

                print(
                    f"[VDN Q] client={client_id}, "
                    f"Q0={q0:+.6f}, "
                    f"Q1={q1:+.6f}, "
                    f"advantage={advantage:+.6f}"
                )



            advantages = (
                q_values_np[:, 1]
                - q_values_np[:, 0]
            )


            effective_buffer_n = (
                vdn_selector.replay_buffer.n
                + (1 if vdn_selector.pending_transition is not None else 0)
            )



            if effective_buffer_n < VDN_WARMUP_TRANSITIONS:
                warmup_rng = random.Random(
                    VDN_SELECTION_SEED + logical_round
                )

                vdn_ranked_ids = warmup_rng.sample(
                    available_client_ids,
                    k=k_selected,
                )

                print(
                    f"[VDN WARMUP] "
                    f"round={logical_round}, "
                    f"buffer_n={effective_buffer_n}, "
                    f"selected={vdn_ranked_ids}"
                )

            else:

                ranking_indices = np.argsort(
                    advantages
                )[::-1]

                ranking_indices = np.array([
                    index for index in ranking_indices
                    if all_client_ids[index] in available_set
                ], dtype=np.int64)

                topk_indices = ranking_indices[
                    :min(
                        k_selected,
                        len(all_client_ids),
                    )
                ]

                vdn_ranked_ids = [
                    all_client_ids[index]
                    for index in topk_indices
                ]

                print(
                    f"[VDN RANK] "
                    f"Top-{k_selected}={vdn_ranked_ids}"
                )



        if selection_mode == "random":
            random_seed = int(os.environ.get("RANDOM_SELECTION_SEED", "4096"))

            rng = random.Random(random_seed + logical_round)

            selected_results = rng.sample(
                results,
                k=min(k_selected, len(results))
                )

            selected_ids = [
                fit_res.metrics.get("client_id", "unknown")
                for _, fit_res in selected_results
                ]

            print(
                f"[RANDOM] Round {logical_round}: "
                f"received={len(results)}, "
                f"selected={len(selected_results)}, "
                f"selected_ids={selected_ids}, "
                f"failures={len(failures)}"
                )

        elif selection_mode == "vdn":
            selected_id_set = set(vdn_ranked_ids)

            selected_results = [
                (client_proxy, fit_res)
                for client_proxy, fit_res in results
                if fit_res.metrics.get(
                    "client_id",
                    "unknown"
                ) in selected_id_set
            ]

            if len(selected_results) != k_selected:
                raise RuntimeError(
                    f"VDN selecionou {len(selected_results)} "
                    f"resultados, mas K_SELECTED={k_selected}."
                )

            selected_ids = vdn_ranked_ids

            print(
                f"[VDN SELECT] Round {logical_round}: "
                f"received={len(results)}, "
                f"selected={len(selected_results)}, "
                f"selected_ids={selected_ids}, "
                f"failures={len(failures)}"
                )
        else:
            raise ValueError(
            f"SELECTION_MODE inválido: {selection_mode}"
            )


        actions = np.zeros(
            len(all_client_ids),
            dtype=np.uint8,
        )

        selected_id_set = set(selected_ids)

        for index, client_id in enumerate(all_client_ids):
            if client_id in selected_id_set:
                actions[index] = 1

        if int(actions.sum()) != k_selected:
            raise RuntimeError(
                f"Vetor de ações possui {int(actions.sum())} "
                f"selecionados, mas K_SELECTED={k_selected}."
            )

        print(
            f"[ACTION] order={all_client_ids}, "
            f"actions={actions.tolist()}"
            )


        update_staleness_streak(
            staleness=client_staleness,
             streak=client_streak,
            all_client_ids=all_client_ids,
            selected_ids=selected_ids,
            )

        print(
            f"[STATE] staleness={client_staleness}, "
            f"streak={client_streak}"
            )


        aggregated_parameters, metrics = super().aggregate_fit(
            server_rounds,
            selected_results,
            failures
        )

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(
                aggregated_parameters
            )

            net = set_weights(
                net,
                ndarrays,
            )

            validation_loss = evaluate_server_loss(
                net=net,
                validation_loader=probe_loader,
                max_batches=20,
            )

            reward = windowed_reward(
                loss_history=validation_loss_history,
                new_loss=validation_loss,
                window_size=REWARD_WINDOW,
            )

            validation_loss_history.append(
                validation_loss
            )

            print(
                f"[REWARD] "
                f"val_loss={validation_loss:.6f}, "
                f"reward={reward:+.6f}, "
                f"history={len(validation_loss_history)}"
            )


        if vdn_selector.pending_transition is not None:
            previous_state, previous_action, previous_reward = (
            vdn_selector.pending_transition
            )

            print(
                f"[TRANSITION READY] "
                f"prev_state_shape={previous_state.shape}, "
                f"prev_action={previous_action.tolist()}, "
                f"prev_reward={previous_reward:+.6f}, "
                f"next_state_shape={state_matrix.shape}"
            )


            vdn_selector.replay_buffer.add(
                obs=previous_state,
                act=previous_action,
                reward=previous_reward,
                obs2=state_matrix.copy(),
                done=False,
            )

            print(
                f"[REPLAY ADD] "
                f"buffer_n={vdn_selector.replay_buffer.n}"
            )

            beta_step = min(
                vdn_selector.train_calls,
                VDN_PER_BETA_STEPS,
            )

            beta_fraction = (
                beta_step
                / max(1, VDN_PER_BETA_STEPS)
            )

            vdn_beta = (
                VDN_PER_BETA_START
                + beta_fraction
                * (
                    VDN_PER_BETA_END
                    - VDN_PER_BETA_START
                )
            )



            if vdn_selector.replay_buffer.n >= VDN_WARMUP_TRANSITIONS:
                (
                sample_obs,
                sample_act,
                sample_reward,
                sample_obs2,
                sample_done,
                sample_indices,
                sample_weights,
            ) = vdn_selector.replay_buffer.sample(
                batch_size=2,
                beta=vdn_beta,
            )

                print(
                    f"[REPLAY SAMPLE] "
                    f"obs={sample_obs.shape}, "
                    f"act={sample_act.shape}, "
                    f"reward={sample_reward.shape}, "
                    f"obs2={sample_obs2.shape}, "
                    f"done={sample_done.shape}, "
                    f"indices={sample_indices.tolist()}, "
                    f"weights={sample_weights.tolist()}"
                )

                batch_count = sample_obs.shape[0]
                agent_count = len(all_client_ids)

                obs_tensor = torch.as_tensor(
                    sample_obs,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                action_tensor = torch.as_tensor(
                    sample_act,
                    dtype=torch.long,
                    device=DEVICE,
                )

                reward_tensor = torch.as_tensor(
                    sample_reward,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                next_obs_tensor = torch.as_tensor(
                    sample_obs2,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                done_tensor = torch.as_tensor(
                    sample_done,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                weight_tensor = torch.as_tensor(
                    sample_weights,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                vdn_selector.q.train()
                vdn_selector.q_tgt.eval()

                current_q = vdn_selector.q(
                    obs_tensor.reshape(
                        batch_count * agent_count,
                        5,
                    )
                ).reshape(
                    batch_count,
                    agent_count,
                    2,
                )

                selected_current_q = current_q.gather(
                    dim=2,
                    index=action_tensor.unsqueeze(2),
                ).squeeze(2)

                current_joint_q = (
                    selected_current_q.sum(dim=1)
                )

                with torch.no_grad():
                    next_q_online = vdn_selector.q(
                        next_obs_tensor.reshape(
                            batch_count * agent_count,
                            5,
                        )
                    ).reshape(
                        batch_count,
                        agent_count,
                        2,
                    )

                    next_q_target = vdn_selector.q_tgt(
                        next_obs_tensor.reshape(
                            batch_count * agent_count,
                            5,
                        )
                    ).reshape(
                        batch_count,
                        agent_count,
                        2,
                    )

                    next_advantages = (
                        next_q_online[:, :, 1]
                        - next_q_online[:, :, 0]
                    )

                    next_available = next_obs_tensor[:, :, 0] > 0.5
                    if torch.any(next_available.sum(dim=1) < k_selected):
                        raise RuntimeError(
                            "Replay contém estado com menos de K disponíveis."
                        )
                    next_advantages = next_advantages.masked_fill(
                        ~next_available, float("-inf")
                    )

                    selected_next_indices = (
                        next_advantages.topk(
                            min(
                                k_selected,
                                agent_count,
                            ),
                            dim=1,
                        ).indices
                    )

                    next_actions = torch.zeros(
                        (batch_count, agent_count),
                        dtype=torch.long,
                        device=DEVICE,
                    )

                    next_actions.scatter_(
                        dim=1,
                        index=selected_next_indices,
                        value=1,
                    )

                    selected_next_q = (
                        next_q_target.gather(
                            dim=2,
                            index=next_actions.unsqueeze(2),
                        ).squeeze(2)
                    )

                    next_joint_q = (
                        selected_next_q.sum(dim=1)
                    )

                    target = (
                        reward_tensor
                        + (1.0 - done_tensor)
                        * VDN_GAMMA
                        * next_joint_q
                    )

                td_error = (
                    current_joint_q - target
                )

                td_error_absolute = (
                    td_error
                    .detach()
                    .abs()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                element_losses = F.smooth_l1_loss(
                    current_joint_q,
                    target,
                    reduction="none",
                )

                loss = (
                    weight_tensor
                    * element_losses
                ).mean()

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "A perda do VDN não é finita."
                    )

                vdn_selector.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    vdn_selector.q.parameters(),
                    VDN_GRAD_CLIP,
                )

                vdn_selector.optimizer.step()

                vdn_selector.replay_buffer.update_priorities(
                    indices=sample_indices,
                    td_error_absolute=td_error_absolute,
                )

                vdn_selector.train_calls += 1

                target_synced = False

                if (
                    vdn_selector.train_calls
                    % VDN_TARGET_SYNC_EVERY
                    == 0
                ):
                    vdn_selector.q_tgt.load_state_dict(
                        vdn_selector.q.state_dict()
                    )
                    vdn_selector.q_tgt.eval()
                    target_synced = True

                print(
                    f"[VDN TRAIN] "
                    f"loss={float(loss.item()):.8f}, "
                    f"td_abs_mean="
                    f"{float(td_error_absolute.mean()):.6f}, "
                    f"beta={vdn_beta:.6f}, "
                    f"updates={vdn_selector.train_calls}, "
                    f"target_sync={target_synced}"
                    )

        total_rounds = int(
            os.environ.get("NUM_ROUNDS", "3")
        )

        is_last_round = (
            logical_round >= total_rounds
        )


        if selection_mode == "vdn" and is_last_round:
            vdn_selector.replay_buffer.add(
                obs=state_matrix.copy(),
                act=actions.copy(),
                reward=float(reward),
                obs2=state_matrix.copy(),
                done=True,
            )

            vdn_selector.pending_transition = None

            print(
                f"[TRANSITION TERMINAL] "
                f"round={logical_round}, "
                f"buffer_n={vdn_selector.replay_buffer.n}, "
                f"done=True"
            )

        elif selection_mode == "vdn":
            vdn_selector.pending_transition = (
                state_matrix.copy(),
                actions.copy(),
                float(reward),
            )

            print(
                f"[TRANSITION PENDING] "
                f"state_shape={state_matrix.shape}, "
                f"action={actions.tolist()}, "
                f"reward={reward:+.6f}"
            )


        try:
            with open(f"/model/{ftdl.task_id}.pickle", "wb") as f:
                f.write(net_to_bytes(net))
        except Exception as e:
            print(f"{e}: Could not save model")
        try:
            training_state = experiment_state_dict(
                logical_round=logical_round,
                client_ids=all_client_ids,
            )

            torch.save(
                training_state,
                f"/model/{ftdl.task_id}_training_state.pt",
            )

            print(
                f"[CHECKPOINT] training state salvo, "
                f"round={logical_round}"
            )

        except Exception as e:
            print(
                f"{e}: Could not save training state"
            )

        if aggregated_parameters is not None:
            registrar_metricas_globais(
                net, logical_round, ultima=is_last_round
            )

        return aggregated_parameters, metrics


def registrar_metricas_globais(net, rodada, ultima=False):
    import copy
    import csv
    import gc
    import time
    from pathlib import Path

    intervalo = int(os.environ.get("GLOBAL_EVAL_EVERY", "10"))
    if rodada != 1 and not ultima and rodada % intervalo != 0:
        return

    inicio = time.perf_counter()
    modelo_avaliacao = None
    resultado = None
    try:
        modelo_avaliacao = copy.deepcopy(net.model)
        resultado = modelo_avaliacao.val(
            data="/data/cityscapes/yolo/data.yaml",
            split="val",
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            half=False,
            augment=False,
            plots=False,
            save=False,
            save_txt=False,
            save_json=False,
            verbose=False,
            project="/model/avaliacoes",
            name=f"round_{rodada:04d}",
            exist_ok=True,
        )

        registro = {
            "round": int(rodada),
            "map50": float(resultado.box.map50),
            "map50_95": float(resultado.box.map),
            "precision": float(resultado.box.mp),
            "recall": float(resultado.box.mr),
            "eval_seconds": time.perf_counter() - inicio,
        }

        destino = Path("/model/metricas_globais.csv")
        cabecalho = not destino.exists() or destino.stat().st_size == 0
        with destino.open("a", newline="") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=list(registro))
            if cabecalho:
                writer.writeheader()
            writer.writerow(registro)

        print(f"[GLOBAL EVAL] {registro}", flush=True)

    except Exception:
        import traceback
        print(
            f"[GLOBAL EVAL ERROR] round={rodada}; "
            "avaliacao falhou, confira o traceback",
            flush=True,
        )
        traceback.print_exc()
    finally:
        del resultado, modelo_avaliacao
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_strategy():
    global net

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)




    return ModFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=int(os.environ["K_SELECTED"]),
        min_evaluate_clients=num_clients,
        min_available_clients=int(os.environ["K_SELECTED"]),
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    return ModFedAvg(
        fraction_fit=1.0,  # Select all available clients
        fraction_evaluate=1.0,  # Disable evaluation
        min_fit_clients=int(os.environ["K_SELECTED"]),
        min_evaluate_clients=num_clients,
        min_available_clients=int(os.environ["K_SELECTED"]),
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

if __name__ == "__main__":

    try:
        ftdl = LocalFTDL(side="server") if LOCAL_TEST else FTDL(side="server")



        num_clients = int(os.environ.get("NUM_CLIENTS", "2"))

        total_rounds = int(
            os.environ.get("NUM_ROUNDS", "5")
        )


        net = Net()

        try:
            with open(f"/model/{ftdl.task_id}.pickle", "rb") as f:
                net = bytes_to_net(net, f.read())
        except Exception as e:
            print(f"{e}: Could not load model from file. Starting from scratch.")



        training_state_path = (
            f"/model/{ftdl.task_id}_training_state.pt"
        )

        try:
            training_state = torch.load(
                training_state_path,
                map_location="cpu",
                weights_only=False,
            )

            saved_client_ids = training_state.get(
                "client_ids",
                [],
            )

            if (
                saved_client_ids
                and len(saved_client_ids) != num_clients
            ):
                raise ValueError(
                    f"Checkpoint possui {len(saved_client_ids)} clientes, "
                    f"mas NUM_CLIENTS={num_clients}."
                )

            completed_rounds = int(
                training_state.get(
                    "completed_rounds",
                    0,
                )
            )

            reference_momentum = training_state.get(
                "reference_momentum"
            )

            client_staleness.clear()
            client_staleness.update(
                training_state.get(
                    "client_staleness",
                    {},
                )
            )

            client_streak.clear()
            client_streak.update(
                training_state.get(
                    "client_streak",
                    {},
                )
            )

            validation_loss_history.clear()
            validation_loss_history.extend(
                training_state.get(
                    "validation_loss_history",
                    [],
                )
            )

            vdn_state = training_state.get("vdn")

            if vdn_state is not None:
                vdn_selector.load_state_dict(
                    vdn_state
                )

            replay_n = (
                vdn_selector.replay_buffer.n
                if vdn_selector.replay_buffer is not None
                else 0
            )

            print(
                f"[CHECKPOINT] training state carregado, "
                f"completed_rounds={completed_rounds}, "
                f"vdn_updates={vdn_selector.train_calls}, "
                f"replay_n={replay_n}"
            )

        except FileNotFoundError:
            print(
                "[CHECKPOINT] training state não encontrado; "
                "iniciando estado da VDN do zero"
            )

        except Exception as e:
            raise RuntimeError(
                f"Could not load training state: {e}"
            ) from e


        remaining_rounds = (
            total_rounds - completed_rounds
        )

        if remaining_rounds <= 0:
            print(
                f"[CHECKPOINT] Experimento já concluído: "
                f"{completed_rounds}/{total_rounds} rodadas."
            )
            raise SystemExit(0)

        print(
            f"[ROUNDS] completed={completed_rounds}, "
            f"remaining={remaining_rounds}, "
            f"target={total_rounds}"
        )

        config = ServerConfig(
            num_rounds=remaining_rounds,
            round_timeout=float(os.environ.get("ROUND_TIMEOUT_SECONDS", "450")),
        )

        probe_loader = load_probe_loader()

        print(
            f"[PROBE] loader do servidor carregado, "
            f"batches={len(probe_loader)}"
            )




        max_workers = int(os.environ.get("MAX_WORKERS", "8"))

        flower_server = Server(
            client_manager=SimpleClientManager(),
            strategy=get_strategy(),
        )

        flower_server.set_max_workers(max_workers)

        print(f"[FLOWER] max_workers={max_workers}")

        start_server(
            server_address="0.0.0.0:8080",
            server=flower_server,
            config=config,
        )


        ftdl.announce_finish(net_to_bytes(net))

    except Exception as e:
        print(e)
        ftdl.announce_error(e)

    print("Waiting for termination...")
    while(True):
        pass

