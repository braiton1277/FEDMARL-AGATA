try:
    import os
    import random
    from typing import List, Tuple

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from flwr.common import (Code, FitIns, GetPropertiesIns, Metrics,
                             ndarrays_to_parameters, parameters_to_ndarrays)
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg
    from ftdl.ftdl import FTDL
    from task import (Net, bytes_to_net, get_weights, load_server_validation,
                      net_to_bytes, server_reference_grad, set_weights)



except Exception as e:
    print("Could not import module:", e)

    ftdl = FTDL(side="server")
    ftdl.announce_error(e)

    print("Waiting for termination...")

    while True:
        pass

VDN_DEVICE = torch.device("cpu")


def flatten_ndarrays(ndarrays):
    if not ndarrays:
        raise ValueError(
            "A lista de parâmetros está vazia."
        )

    return np.concatenate(
        [
            np.asarray(array, dtype=np.float32).reshape(-1)
            for array in ndarrays
        ],
        axis=0,
    )





def weighted_average(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    valid_metrics = [
        (num_examples, client_metrics)
        for num_examples, client_metrics in metrics
        if "metric" in client_metrics
    ]

    if not valid_metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in valid_metrics
    )

    if total_examples == 0:
        return {"metric": 0.0}

    weighted_loss = sum(
        num_examples * float(client_metrics["metric"])
        for num_examples, client_metrics in valid_metrics
    )

    result = {
        "metric": weighted_loss / total_examples
    }

    print(
        f"Métrica agregada: {result}",
        flush=True,
    )

    return result


@torch.no_grad()
def evaluate_server_loss(
    model,
    validation_loader,
    max_batches: int = 20,
) -> float:
    if max_batches <= 0:
        raise ValueError(
            "max_batches deve ser maior que zero."
        )

    was_training = model.training
    model.eval()

    device = next(
        model.parameters()
    ).device

    total_loss = 0.0
    batches_used = 0

    for batch_index, (
        inputs,
        targets,
    ) in enumerate(validation_loader):

        if batch_index >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)

        loss = F.cross_entropy(
            predictions,
            targets,
        )

        total_loss += float(
            loss.item()
        )

        batches_used += 1

    if was_training:
        model.train()

    if batches_used == 0:
        raise RuntimeError(
            "Nenhum lote foi usado na avaliação."
        )

    average_loss = (
        total_loss / batches_used
    )

    if not np.isfinite(average_loss):
        raise RuntimeError(
            "A perda de validação não é finita."
        )

    return float(average_loss)


def windowed_reward(
    loss_history,
    new_loss: float,
    window_size: int = 5,
) -> float:
    if window_size <= 0:
        raise ValueError(
            "window_size deve ser maior que zero."
        )

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




def build_context_matrix_vdn(
    projection_mom: np.ndarray,
    probe_now: np.ndarray,
    staleness: np.ndarray,
    streak: np.ndarray,
) -> np.ndarray:
    projection = projection_mom.astype(
        np.float32
    )

    probe = probe_now.astype(
        np.float32
    )

    stale = staleness.astype(
        np.float32
    )

    stale_max = float(
        stale.max() + 1e-6
    )

    staleness_normalized = (
        stale / stale_max
    ).astype(np.float32)

    streak_cap = 5.0

    streak_float = streak.astype(
        np.float32
    )

    streak_normalized = np.clip(
        streak_float / streak_cap,
        0.0,
        1.0,
    ).astype(np.float32)

    bias = np.ones(
        projection.shape[0],
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


def update_staleness_streak(
    staleness: np.ndarray,
    streak: np.ndarray,
    selected_ids,
):
    selected_mask = np.zeros(
        staleness.shape[0],
        dtype=bool,
    )

    selected_mask[selected_ids] = True

    staleness[~selected_mask] += 1.0
    staleness[selected_mask] = 0.0

    streak[~selected_mask] = 0
    streak[selected_mask] += 1



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

        if self.capacity <= 0:
            raise ValueError(
                "capacity deve ser maior que zero."
            )

        self.obs = np.zeros(
            (
                self.capacity,
                self.n_agents,
                self.d_in,
            ),
            dtype=np.float32,
        )

        self.act = np.zeros(
            (
                self.capacity,
                self.n_agents,
            ),
            dtype=np.uint8,
        )

        self.reward = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.obs2 = np.zeros(
            (
                self.capacity,
                self.n_agents,
                self.d_in,
            ),
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
        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        act = np.asarray(
            act,
            dtype=np.uint8,
        )

        obs2 = np.asarray(
            obs2,
            dtype=np.float32,
        )

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
                f"mas era esperado "
                f"{expected_obs_shape}."
            )

        if obs2.shape != expected_obs_shape:
            raise ValueError(
                f"obs2 possui dimensão {obs2.shape}, "
                f"mas era esperado "
                f"{expected_obs_shape}."
            )

        if act.shape != expected_action_shape:
            raise ValueError(
                f"act possui dimensão {act.shape}, "
                f"mas era esperado "
                f"{expected_action_shape}."
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
            probabilities = np.ones_like(
                probabilities
            ) / len(probabilities)
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





class AgentMLP(nn.Module):

    def __init__(
        self,
        d_in: int,
        hidden: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class VDNSelector:

    def __init__(
        self,
        n_agents: int,
        d_in: int,
        k_select: int,
        hidden: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        buf_size: int = 20000,
        per_alpha: float = 0.6,
        per_eps: float = 1e-3,
        seed: int = 2048,
    ):
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.k_select = int(k_select)

        if self.n_agents <= 0:
            raise ValueError(
                "n_agents deve ser maior que zero."
            )

        if self.d_in <= 0:
            raise ValueError(
                "d_in deve ser maior que zero."
            )

        if not 1 <= self.k_select <= self.n_agents:
            raise ValueError(
                "k_select deve estar entre 1 e n_agents."
            )

        
        rng_state = torch.random.get_rng_state()

        torch.manual_seed(int(seed))

        self.q = AgentMLP(
            d_in=self.d_in,
            hidden=hidden,
        ).to(VDN_DEVICE)

        self.q_tgt = AgentMLP(
            d_in=self.d_in,
            hidden=hidden,
        ).to(VDN_DEVICE)

        self.q_tgt.load_state_dict(
            self.q.state_dict()
        )

        torch.random.set_rng_state(rng_state)

        self.opt = torch.optim.Adam(
            self.q.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )



        self.buf = PrioritizedReplayJoint(
            capacity=buf_size,
            n_agents=self.n_agents,
            d_in=self.d_in,
            alpha=per_alpha,
            eps=per_eps,
            seed=int(seed) + 12345,
        )



        self.py_rng = random.Random(
            int(seed) + 777
        )

        self.np_rng = np.random.default_rng(
            int(seed) + 999
        )

    @torch.no_grad()
    def q_values(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        expected_shape = (
            self.n_agents,
            self.d_in,
        )

        if obs.shape != expected_shape:
            raise ValueError(
                f"O estado possui dimensão {obs.shape}, "
                f"mas era esperado {expected_shape}."
            )

        self.q.eval()

        state_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=VDN_DEVICE,
        )

        q_values = self.q(
            state_tensor
        )

        return (
            q_values
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def select_topk_actions(
        self,
        obs: np.ndarray,
        eps: float = 0.0,
        swap_m: int = 0,
        force_random: bool = False,
    ):
        q_values = self.q_values(obs)

        advantages = (
            q_values[:, 1]
            - q_values[:, 0]
        )

        n_agents = obs.shape[0]
        k_select = min(
            self.k_select,
            n_agents,
        )

        if force_random:
            selected = self.py_rng.sample(
                range(n_agents),
                k_select,
            )

        else:
            selected = (
                np.argsort(advantages)[::-1]
                [:k_select]
                .tolist()
            )

            perform_swap = (
                swap_m > 0
                and self.np_rng.random() < eps
            )

            if perform_swap:
                swap_count = min(
                    swap_m,
                    k_select,
                )

                selected_set = set(selected)

                not_selected = [
                    client_id
                    for client_id in range(n_agents)
                    if client_id not in selected_set
                ]

                if not_selected:
                    outgoing = self.py_rng.sample(
                        selected,
                        swap_count,
                    )

                    incoming = self.py_rng.sample(
                        not_selected,
                        min(
                            swap_count,
                            len(not_selected),
                        ),
                    )

                    new_selected = selected.copy()

                    for old_id, new_id in zip(
                        outgoing,
                        incoming,
                    ):
                        position = new_selected.index(
                            old_id
                        )

                        new_selected[position] = (
                            new_id
                        )

                    selected = list(
                        dict.fromkeys(new_selected)
                    )

                    while len(selected) < k_select:
                        candidate_id = (
                            self.py_rng.randrange(
                                n_agents
                            )
                        )

                        if candidate_id not in selected:
                            selected.append(
                                candidate_id
                            )

                    selected = selected[
                        :k_select
                    ]

        actions = np.zeros(
            n_agents,
            dtype=np.uint8,
        )

        actions[selected] = 1

        return (
            actions,
            selected,
            q_values,
            advantages,
        )

    def add_transition(
        self,
        obs,
        act,
        reward,
        obs2,
        done: bool,
    ):
        self.buf.add(
            obs=obs,
            act=act,
            reward=reward,
            obs2=obs2,
            done=done,
        )





class ModFedAvg(FedAvg):

    def __init__(
        self,
        n_clients: int,
        total_rounds: int,
        k_select: int,
        selection_seed: int,
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.total_rounds = int(
            total_rounds
        )

        if self.total_rounds <= 0:
            raise ValueError(
                "total_rounds deve ser maior que zero."
            )



        self.n_clients = int(n_clients)

        if self.n_clients <= 0:
            raise ValueError(
                "n_clients deve ser maior que zero."
            )

        self.k_select = int(k_select)
        self.selection_seed = int(selection_seed)

        self.client_id_by_proxy = {}
        self.momentum_beta = 0.90
        self.reference_momentum = None
        self.staleness = np.zeros(
            self.n_clients,
            dtype=np.float32,
            )

        self.streak = np.zeros(
            self.n_clients,
            dtype=np.int32,
         )


        self.vdn_selector = VDNSelector(
            n_agents=self.n_clients,
            d_in=5,
            k_select=self.k_select,
            hidden=128,
            lr=1e-3,
            weight_decay=1e-4,
            seed=self.selection_seed,
        )

        self.reward_window = 5

        self.validation_loss_history = []

        self.pending_transition = None

        
        self.vdn_epsilon = 0.0
        self.vdn_swap_m = 0




    def register_connected_clients(
    self,
    server_round,
    client_manager,
    ):
        
        print(
            f"[Rodada {server_round}] "
            f"aguardando {self.min_available_clients} clientes...",
            flush=True,
        )

        clients_available = client_manager.wait_for(
            num_clients=self.min_available_clients,
            timeout=300,
        )

        if not clients_available:
            raise TimeoutError(
                f"Tempo esgotado aguardando "
                f"{self.min_available_clients} clientes. "
                f"Disponíveis: {client_manager.num_available()}."
            )

        print(
            f"[Rodada {server_round}] "
            f"{client_manager.num_available()} clientes conectados.",
            flush=True,
        )

        connected_clients = list(
            client_manager.all().values()
        )

        registered_clients = []
        client_ids_found = set()

        for client_proxy in connected_clients:
            response = client_proxy.get_properties(
                ins=GetPropertiesIns(
                    config={"request": "identity"}
                ),
                timeout=30.0,
                group_id=server_round,
            )

            if response.status.code != Code.OK:
                raise RuntimeError(
                    "Não foi possível obter as propriedades "
                    f"do cliente Flower {client_proxy.cid}: "
                    f"{response.status.message}"
                )

            properties = response.properties

            if "client_id" not in properties:
                raise RuntimeError(
                    f"O cliente Flower {client_proxy.cid} "
                    "não informou client_id."
                )

            client_id = int(
                properties["client_id"]
            )

            if client_id in client_ids_found:
                raise RuntimeError(
                    f"CLIENT_ID duplicado: {client_id}."
                )

            client_ids_found.add(client_id)

            self.client_id_by_proxy[
                client_proxy.cid
            ] = client_id

            num_examples = int(
                properties.get("num_examples", 0)
            )

            registered_clients.append(
                (
                    client_id,
                    client_proxy,
                    num_examples,
                )
            )

            print(
                f"[Registro] Flower CID={client_proxy.cid} "
                f"-> CLIENT_ID={client_id}, "
                f"amostras={num_examples}",
                flush=True,
            )

        registered_clients.sort(
            key=lambda item: item[0]
        )

        return registered_clients


    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
        ):

        registered_clients = (
            self.register_connected_clients(
                server_round=server_round,
                client_manager=client_manager,
            )
        )

        base_config = {}

        if self.on_fit_config_fn is not None:
            base_config = self.on_fit_config_fn(
                server_round
            )

        instructions = []

        for (
            client_id,
            client_proxy,
            _num_examples,
        ) in registered_clients:
            client_config = dict(base_config)

            client_config["server_round"] = int(
                server_round
            )

            client_config["client_id"] = int(
                client_id
            )

            fit_instruction = FitIns(
                parameters=parameters,
                config=client_config,
            )

            instructions.append(
                (
                    client_proxy,
                    fit_instruction,
                )
            )

        available_ids = [
            client_id
            for client_id, _, _ in registered_clients
        ]

        print(
            f"[Rodada {server_round}] "
            f"clientes candidatos={available_ids}",
            flush=True,
        )

        print(
            f"[Rodada {server_round}] "
            "enviando treinamento para todos os candidatos",
            flush=True,
        )

        return instructions
    
    def aggregate_fit(
    self,
    server_round,
    results,
    failures,
):
        global net

        print(
            f"[Rodada {server_round}] "
            f"atualizações candidatas={len(results)}, "
            f"falhas={len(failures)}",
            flush=True,
        )

        candidate_results = []  
        global_ndarrays = get_weights(net)

        global_vector = flatten_ndarrays(
            global_ndarrays
        )


        reference_gradient = server_reference_grad(
            model=net,
            validation_loader=server_val_loader,
            batches=10,
        )

        if reference_gradient.shape != global_vector.shape:
            raise RuntimeError(
                f"O gradiente de referência possui "
                f"{reference_gradient.size} elementos, "
                f"mas o modelo global possui "
                f"{global_vector.size}."
            )




        reference_gradient_norm = float(
            np.linalg.norm(
                reference_gradient
            )
        )

        if not np.isfinite(
            reference_gradient_norm
        ):
            raise RuntimeError(
                "A norma do gradiente de referência "
                "não é finita."
            )

        if reference_gradient_norm <= 0.0:
            raise RuntimeError(
                "O gradiente de referência é nulo."
            )


        if self.reference_momentum is None:
            self.reference_momentum = (
                reference_gradient.copy()
            )
        else:
            if (
                self.reference_momentum.shape
                != reference_gradient.shape
            ):
                raise RuntimeError(
                    "O momento do gradiente possui "
                    "dimensão incompatível."
                )

            self.reference_momentum = (
                self.momentum_beta
                * self.reference_momentum
                + (1.0 - self.momentum_beta)
                * reference_gradient
            )

        momentum_norm = float(
            np.linalg.norm(
                self.reference_momentum
            )
        )

        if not np.isfinite(momentum_norm):
            raise RuntimeError(
                "A norma do momento do gradiente "
                "não é finita."
            )

        if momentum_norm <= 0.0:
            raise RuntimeError(
                "O momento do gradiente é nulo."
            )

        descent_momentum_direction = (
            -self.reference_momentum
            / momentum_norm
        )

        print(
            f"[Rodada {server_round}] "
            f"momento_beta={self.momentum_beta}, "
            f"momento_norm={momentum_norm:.6f}",
            flush=True,
        )




        print(
            f"[Rodada {server_round}] "
            f"gradiente de referência="
            f"{reference_gradient.size}, "
            f"norma={reference_gradient_norm:.6f}",
            flush=True,
        )



        print(
            f"[Rodada {server_round}] "
            f"parâmetros globais={global_vector.size}",
            flush=True,
        )

        for client_proxy, fit_result in results:
            registered_client_id = (
                self.client_id_by_proxy.get(
                    client_proxy.cid
                )
            )

            returned_client_id = (
                fit_result.metrics.get("client_id")
            )

            if registered_client_id is None:
                raise RuntimeError(
                    f"O proxy Flower {client_proxy.cid} "
                    "não possui CLIENT_ID registrado."
                )

            if returned_client_id is None:
                raise RuntimeError(
                    f"O cliente {registered_client_id} "
                    "não retornou client_id nas métricas."
                )

            if int(returned_client_id) != int(
                registered_client_id
            ):
                raise RuntimeError(
                    "Identidade inconsistente: "
                    f"proxy {client_proxy.cid} foi registrado "
                    f"como cliente {registered_client_id}, "
                    f"mas retornou {returned_client_id}."
                )

            probe_value = fit_result.metrics.get(
                "probe_loss"
            )


            if probe_value is None:
                raise RuntimeError(
                    f"O cliente {registered_client_id} "
                    "não retornou probe_loss."
                )


            candidate_ndarrays = parameters_to_ndarrays(
                fit_result.parameters
            )

            candidate_vector = flatten_ndarrays(
                candidate_ndarrays
            )

            if candidate_vector.shape != global_vector.shape:
                raise RuntimeError(
                    f"O cliente {registered_client_id} retornou "
                    f"{candidate_vector.size} parâmetros, mas o "
                    f"modelo global possui {global_vector.size}."
                )

            client_delta = (
                candidate_vector - global_vector
            )

            delta_norm = float(
                np.linalg.norm(client_delta)
            )


            projection_mom = float(
                np.dot(
                    client_delta,
                    descent_momentum_direction,
                )
            )

            if not np.isfinite(projection_mom):
                raise RuntimeError(
                    f"A projeção do cliente "
                    f"{registered_client_id} "
                    "não é finita."
                )




            train_loss = fit_result.metrics.get(
                "metric"
            )

            candidate_results.append(
                (
                    int(registered_client_id),
                    client_proxy,
                    fit_result,
                    client_delta,
                    projection_mom,
                )
            )

            print(
                f"[Candidato] CLIENT_ID="
                f"{registered_client_id}, "
                f"amostras={fit_result.num_examples}, "
                f"probe_loss={probe_value}, "
                f"train_loss={train_loss}, "
                f"delta_norm={delta_norm:.6f}, "
                f"projection_mom={projection_mom:+.6f}",
                flush=True,
                )

        candidate_results.sort(
            key=lambda item: item[0]
        )


        received_ids = [
            client_id
            for client_id, _, _, _, _
            in candidate_results
        ]

        expected_ids = list(
            range(self.n_clients)
        )

        if received_ids != expected_ids:
            raise RuntimeError(
                f"IDs recebidos={received_ids}, "
                f"mas eram esperados={expected_ids}."
            )

        projection_by_client = np.zeros(
            self.n_clients,
            dtype=np.float32,
        )

        probe_by_client = np.zeros(
            self.n_clients,
            dtype=np.float32,
        )

        for (
            client_id,
            _client_proxy,
            fit_result,
            _client_delta,
            projection_mom,
        ) in candidate_results:
            projection_by_client[client_id] = float(
                projection_mom
            )

            probe_by_client[client_id] = float(
                fit_result.metrics["probe_loss"]
            )

        state_matrix = build_context_matrix_vdn(
            projection_mom=projection_by_client,
            probe_now=probe_by_client,
            staleness=self.staleness,
            streak=self.streak,
        )

        if state_matrix.shape != (
            self.n_clients,
            5,
        ):
            raise RuntimeError(
                f"O estado possui dimensão "
                f"{state_matrix.shape}, mas deveria "
                f"ser ({self.n_clients}, 5)."
            )

        for client_id in range(self.n_clients):
            state = state_matrix[client_id]

            print(
                f"[Estado] CLIENT_ID={client_id}, "
                f"state=["
                f"{state[0]:.3f}, "
                f"{state[1]:+.6f}, "
                f"{state[2]:.6f}, "
                f"{state[3]:.3f}, "
                f"{state[4]:.3f}]",
                flush=True,
            )


        if self.pending_transition is not None:
            (
                previous_state,
                previous_action,
                previous_reward,
            ) = self.pending_transition

            self.vdn_selector.add_transition(
                obs=previous_state,
                act=previous_action,
                reward=previous_reward,
                obs2=state_matrix,
                done=False,
            )

            print(
                f"[Replay] transição anterior "
                f"adicionada, "
                f"buffer_n={self.vdn_selector.buf.n}",
                flush=True,
            )

            self.pending_transition = None




        if self.k_select <= 0:
            raise ValueError(
                "k_select deve ser maior que zero."
            )

        if len(candidate_results) < self.k_select:
            raise RuntimeError(
                f"Foram recebidas apenas "
                f"{len(candidate_results)} atualizações, "
                f"mas são necessárias {self.k_select}."
            )

        (
        actions,
            selected_ranked_ids,
            q_values,
            advantages,
        ) = self.vdn_selector.select_topk_actions(
            obs=state_matrix,
            eps=self.vdn_epsilon,
            swap_m=self.vdn_swap_m,
            force_random=False,
        )

        if int(actions.sum()) != self.k_select:
            raise RuntimeError(
                f"O VDN selecionou {int(actions.sum())} "
                f"clientes, mas deveria selecionar "
                f"{self.k_select}."
            )

        for client_id in range(self.n_clients):
            print(
                f"[VDN] CLIENT_ID={client_id}, "
                f"Q0={q_values[client_id, 0]:+.6f}, "
                f"Q1={q_values[client_id, 1]:+.6f}, "
                f"advantage={advantages[client_id]:+.6f}, "
                f"action={int(actions[client_id])}",
                flush=True,
            )

        print(
            f"[Rodada {server_round}] "
            f"ranking Top-K do VDN="
            f"{selected_ranked_ids}",
            flush=True,
        )

        
        selected_ids = sorted(
            selected_ranked_ids
        )

        candidate_by_id = {
            candidate[0]: candidate
            for candidate in candidate_results
        }


        selected_candidates = [
            candidate_by_id[client_id]
            for client_id in selected_ids
        ]

        print(
            f"[Rodada {server_round}] "
            f"atualizações selecionadas={selected_ids}",
            flush=True,
        )


        update_staleness_streak(
        staleness=self.staleness,
        streak=self.streak,
        selected_ids=selected_ids,
         )

        print(
            f"[Rodada {server_round}] "
            f"staleness={self.staleness.tolist()}, "
            f"streak={self.streak.tolist()}",
            flush=True,
         )


        
        selected_results = [
            (client_proxy, fit_result)
            for (
                _,
                client_proxy,
                fit_result,
                _,
                _,
            ) in selected_candidates
            ]

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                selected_results,
                failures,
            )
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
                model=net,
                validation_loader=server_val_loader,
                max_batches=20,
            )

            reward = windowed_reward(
                loss_history=self.validation_loss_history,
                new_loss=validation_loss,
                window_size=self.reward_window,
            )

            self.validation_loss_history.append(
                validation_loss
            )

            print(
                f"[Recompensa] "
                f"val_loss={validation_loss:.6f}, "
                f"reward={reward:+.6f}, "
                f"historico="
                f"{len(self.validation_loss_history)}",
                flush=True,
            )

            is_last_round = (
                server_round >= self.total_rounds
            )

            if is_last_round:
                self.vdn_selector.add_transition(
                    obs=state_matrix.copy(),
                    act=actions.copy(),
                    reward=reward,
                    obs2=state_matrix.copy(),
                    done=True,
                )

                self.pending_transition = None

                print(
                    f"[Replay] transição terminal "
                    f"adicionada, "
                    f"buffer_n={self.vdn_selector.buf.n}",
                    flush=True,
                )

            else:
                self.pending_transition = (
                    state_matrix.copy(),
                    actions.copy(),
                    float(reward),
                )

                print(
                    f"[Replay] transição atual "
                    f"mantida como pendente, "
                    f"buffer_n={self.vdn_selector.buf.n}",
                    flush=True,
                )



            try:
                model_path = (
                    f"/model/{ftdl.task_id}.pickle"
                )

                with open(model_path, "wb") as file:
                    file.write(
                        net_to_bytes(net)
                    )

                print(
                    f"Modelo global salvo em "
                    f"{model_path}",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"{e}: Could not save model",
                    flush=True,
                )

        return aggregated_parameters, metrics


def get_strategy():
    ndarrays = get_weights(net)

    parameters = ndarrays_to_parameters(
        ndarrays
    )

    return ModFedAvg(
        n_clients=num_clients,
        total_rounds=num_rounds,
        k_select=select_clients,
        selection_seed=selection_seed,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        )


if __name__ == "__main__":

    try:
        ftdl = FTDL(side="server")

        # Neste teste, apenas dois dos 50 clientes
        # estarão conectados.
        num_clients = int(
            os.environ.get(
                "NUM_CLIENTS",
                "2",
            )
        )

        num_rounds = int(
            os.environ.get(
                "NUM_ROUNDS",
                "1",
            )
        )


        select_clients = int(
            os.environ.get(
                "SELECT_CLIENTS",
                "1",
            )
        )

        selection_seed = int(
            os.environ.get(
                "SELECTION_SEED",
                "2048",
            )
        )

        if select_clients > num_clients:
            raise ValueError(
                f"SELECT_CLIENTS={select_clients} não pode ser "
                f"maior que NUM_CLIENTS={num_clients}."
            )

        print(
            f"Configuração: disponíveis={num_clients}, "
            f"selecionados={select_clients}, "
            f"seed={selection_seed}",
            flush=True,
        )


        config = ServerConfig(
            num_rounds=num_rounds
        )

        net = Net()

        server_val_loader = load_server_validation(
            data_path="/data",
            seed=2048,
            per_class=200,
            batch_size=256,
        )

        try:
            model_path = (
                f"/model/{ftdl.task_id}.pickle"
            )

            with open(model_path, "rb") as file:
                net = bytes_to_net(
                    net,
                    file.read(),
                )

            print(
                f"Modelo carregado de {model_path}",
                flush=True,
            )

        except Exception as e:
            print(
                f"{e}: Could not load model from file. "
                "Starting from scratch.",
                flush=True,
            )

        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=get_strategy(),
        )

        ftdl.announce_finish()

    except Exception as e:
        print(e)

        ftdl.announce_error(e)

    print("Waiting for termination...")

    while True:
        pass