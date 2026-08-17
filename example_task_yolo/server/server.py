try:
    import os
    import torch.nn.functional as F
    from typing import List, Tuple
    import random
    import numpy as np
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

vdn_q = None


vdn_q_tgt = None
vdn_optimizer = None
vdn_train_calls = 0

VDN_GAMMA = 0.90
VDN_GRAD_CLIP = 1.0
VDN_TARGET_SYNC_EVERY = 2


pending_transition = None



replay_buffer = None

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
            client_projections[client_id]
            for client_id in client_ids
        ],
        dtype=np.float32,
    )

    probe = np.array(
        [
            client_probe_losses[client_id]
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

    bias = np.ones(
        len(client_ids),
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








class ModFedAvg(FedAvg):
    def aggregate_fit(self, server_rounds, results, failures):
        global net, comm_round, reference_momentum, vdn_q
        global validation_loss_history, pending_transition
        global replay_buffer, vdn_q_tgt, vdn_optimizer, vdn_train_calls
        comm_round = server_rounds

        all_client_ids = sorted([
            fit_res.metrics.get("client_id", "unknown")
            for _, fit_res in results])


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



        state_matrix = build_context_matrix_vdn(
            client_ids=all_client_ids,
            client_projections=client_projections,
            client_probe_losses=client_probe_losses,
            staleness=client_staleness,
            streak=client_streak,
            )


        if replay_buffer is None:
            replay_buffer = PrioritizedReplayJoint(
                capacity=20000,
                n_agents=len(all_client_ids),
                d_in=5,
                alpha=0.6,
                eps=1e-3,
                seed=4096,
            )

            print(
                f"[REPLAY] criado: "
                f"capacity={replay_buffer.capacity}, "
                f"n_agents={replay_buffer.n_agents}, "
                f"d_in={replay_buffer.d_in}"
            )




        for index, client_id in enumerate(all_client_ids):
            print(
                f"[VDN STATE] client={client_id}, "
                f"state={state_matrix[index].tolist()}"
                )




        if vdn_q is None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(4096)

            vdn_q = AgentMLP(
                d_in=5,
                hidden=128,
            ).to(DEVICE)

            vdn_q_tgt = AgentMLP(
                d_in=5,
                hidden=128,
            ).to(DEVICE)

            vdn_q_tgt.load_state_dict(
                vdn_q.state_dict()
            )

            vdn_q_tgt.eval()

            vdn_optimizer = torch.optim.Adam(
                vdn_q.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
            )

            torch.random.set_rng_state(
                rng_state
            )

            print("[VDN INIT] q + q_tgt + optimizer criados")


        vdn_q.eval()

        with torch.no_grad():
            state_tensor = torch.tensor(
                state_matrix,
                dtype=torch.float32,
                device=DEVICE,
            )

            q_values = vdn_q(state_tensor)

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


        k_selected = int(
            os.environ.get("K_SELECTED", "2")
        )

        advantages = (
            q_values_np[:, 1]
            - q_values_np[:, 0]
        )

        ranking_indices = np.argsort(
            advantages
        )[::-1]

        topk_indices = ranking_indices[
            :min(k_selected, len(all_client_ids))
        ]

        vdn_ranked_ids = [
            all_client_ids[index]
            for index in topk_indices
        ]

        print(
            f"[VDN RANK] Top-{k_selected}="
            f"{vdn_ranked_ids}"
        )



        selection_mode = os.environ.get("SELECTION_MODE", "random").lower()


        if selection_mode == "random":
            k_selected = int(os.environ.get("K_SELECTED", "2"))
            random_seed = int(os.environ.get("RANDOM_SELECTION_SEED", "4096"))

            rng = random.Random(random_seed + server_rounds)

            selected_results = rng.sample(
                results,
                k=min(k_selected, len(results))
                )

            selected_ids = [
                fit_res.metrics.get("client_id", "unknown")
                for _, fit_res in selected_results
                ]

            print(
                f"[RANDOM] Round {server_rounds}: "
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
                f"[VDN SELECT] Round {server_rounds}: "
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


        if pending_transition is not None:
            previous_state, previous_action, previous_reward = (
            pending_transition
            )

            print(
                f"[TRANSITION READY] "
                f"prev_state_shape={previous_state.shape}, "
                f"prev_action={previous_action.tolist()}, "
                f"prev_reward={previous_reward:+.6f}, "
                f"next_state_shape={state_matrix.shape}"
            )


            replay_buffer.add(
                obs=previous_state,
                act=previous_action,
                reward=previous_reward,
                obs2=state_matrix.copy(),
                done=False,
            )

            print(
                f"[REPLAY ADD] "
                f"buffer_n={replay_buffer.n}"
            )


            if replay_buffer.n >= 2:
                (
                sample_obs,
                sample_act,
                sample_reward,
                sample_obs2,
                sample_done,
                sample_indices,
                sample_weights,
            ) = replay_buffer.sample(
                batch_size=2,
                beta=0.4,
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

                vdn_q.train()
                vdn_q_tgt.eval()

                current_q = vdn_q(
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
                    next_q_online = vdn_q(
                        next_obs_tensor.reshape(
                            batch_count * agent_count,
                            5,
                        )
                    ).reshape(
                        batch_count,
                        agent_count,
                        2,
                    )

                    next_q_target = vdn_q_tgt(
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

                vdn_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    vdn_q.parameters(),
                    VDN_GRAD_CLIP,
                )

                vdn_optimizer.step()

                replay_buffer.update_priorities(
                    indices=sample_indices,
                    td_error_absolute=td_error_absolute,
                )

                vdn_train_calls += 1

                target_synced = False

                if (
                    vdn_train_calls
                    % VDN_TARGET_SYNC_EVERY
                    == 0
                ):
                    vdn_q_tgt.load_state_dict(
                        vdn_q.state_dict()
                    )
                    vdn_q_tgt.eval()
                    target_synced = True

                print(
                    f"[VDN TRAIN] "
                    f"loss={float(loss.item()):.8f}, "
                    f"td_abs_mean="
                    f"{float(td_error_absolute.mean()):.6f}, "
                    f"updates={vdn_train_calls}, "
                    f"target_sync={target_synced}"
                )



        total_rounds = int(
            os.environ.get("NUM_ROUNDS", "3")
        )

        is_last_round = (
            server_rounds >= total_rounds
        )

        if is_last_round:
            replay_buffer.add(
                obs=state_matrix.copy(),
                act=actions.copy(),
                reward=float(reward),
                obs2=state_matrix.copy(),
                done=True,
            )

            pending_transition = None

            print(
                f"[TRANSITION TERMINAL] "
                f"round={server_rounds}, "
                f"buffer_n={replay_buffer.n}, "
                f"done=True"
            )

        else:
            pending_transition = (
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



        probe_loader = load_probe_loader()

        print(
            f"[PROBE] loader do servidor carregado, "
            f"batches={len(probe_loader)}"
            )


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

