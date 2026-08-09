try:
    import json
    import os
    import random
    from typing import List, Optional, Tuple

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
def evaluate_server_metrics(
    model,
    validation_loader,
    max_batches: int = 20,
) -> tuple[float, float]:
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
    total_correct = 0
    total_samples = 0
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

        predicted_classes = predictions.argmax(
            dim=1
        )

        total_correct += int(
            (
                predicted_classes == targets
            ).sum().item()
        )

        total_samples += int(
            targets.size(0)
        )

        batches_used += 1

    if was_training:
        model.train()

    if batches_used == 0 or total_samples == 0:
        raise RuntimeError(
            "Nenhum lote foi usado na avaliação."
        )

    average_loss = (
        total_loss / batches_used
    )

    accuracy = (
        total_correct / total_samples
    )

    if not np.isfinite(average_loss):
        raise RuntimeError(
            "A perda de validação não é finita."
        )

    if not np.isfinite(accuracy):
        raise RuntimeError(
            "A acurácia de validação não é finita."
        )

    return (
        float(average_loss),
        float(accuracy),
    )


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


def dynamic_batch_size(
    buffer_size: int,
    base: int = 32,
    maximum: int = 256,
    ratio: int = 4,
) -> int:
    batch_size = int(base)
    maximum = int(maximum)
    ratio = int(ratio)

    if batch_size <= 0:
        raise ValueError(
            "base deve ser maior que zero."
        )

    if maximum < batch_size:
        raise ValueError(
            "maximum não pode ser menor que base."
        )

    if ratio <= 0:
        raise ValueError(
            "ratio deve ser maior que zero."
        )

    while (
        batch_size < maximum
        and buffer_size >= ratio * batch_size
    ):
        batch_size *= 2

    return min(batch_size, maximum)


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
        gamma: float = 0.90,
        grad_clip: float = 1.0,
        target_sync_every: int = 20,
        buf_size: int = 20000,
        batch_size: int = 128,
        train_steps: int = 20,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_steps: int = 4000,
        per_eps: float = 1e-3,
        double_dqn: bool = True,
        seed: int = 2048,
    ):
        self.n_agents = int(n_agents)
        self.d_in = int(d_in)
        self.k_select = int(k_select)

        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.target_sync_every = int(
            target_sync_every
        )
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.double_dqn = bool(double_dqn)

        self.per_beta_start = float(
            per_beta_start
        )
        self.per_beta_end = float(
            per_beta_end
        )
        self.per_beta_steps = int(
            per_beta_steps
        )

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

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "gamma deve estar entre zero e um."
            )

        if self.target_sync_every <= 0:
            raise ValueError(
                "target_sync_every deve ser maior que zero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size deve ser maior que zero."
            )

        if self.train_steps <= 0:
            raise ValueError(
                "train_steps deve ser maior que zero."
            )

        if self.per_beta_steps <= 0:
            raise ValueError(
                "per_beta_steps deve ser maior que zero."
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

        self.q_tgt.eval()

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

        self._train_calls = 0
        self.total_updates = 0
        self.total_samples_drawn = 0

        self.last_train_beta = None
        self.last_td_abs_mean = None
        self.last_target_sync = False

        self.py_rng = random.Random(
            int(seed) + 777
        )

        self.np_rng = np.random.default_rng(
            int(seed) + 999
        )

    def _beta(self) -> float:
        step = min(
            self._train_calls,
            self.per_beta_steps,
        )

        fraction = (
            step / max(1, self.per_beta_steps)
        )

        return (
            self.per_beta_start
            + fraction
            * (
                self.per_beta_end
                - self.per_beta_start
            )
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

    def train(
        self,
        batch_size: Optional[int] = None,
        train_steps: Optional[int] = None,
    ) -> Optional[float]:
        requested_batch_size = (
            int(batch_size)
            if batch_size is not None
            else self.batch_size
        )

        steps = (
            int(train_steps)
            if train_steps is not None
            else self.train_steps
        )

        if requested_batch_size <= 0:
            raise ValueError(
                "batch_size deve ser maior que zero."
            )

        if steps <= 0:
            raise ValueError(
                "train_steps deve ser maior que zero."
            )

        if self.buf.n < requested_batch_size:
            return None

        beta = self._beta()
        self.last_train_beta = float(beta)
        self.last_target_sync = False

        self.q.train()
        self.q_tgt.eval()

        losses = []
        td_means = []

        for _ in range(steps):
            (
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                indices,
                importance_weights,
            ) = self.buf.sample(
                batch_size=requested_batch_size,
                beta=beta,
            )

            batch_count = observations.shape[0]
            agent_count = self.n_agents

            self.total_samples_drawn += (
                batch_count
            )

            obs_tensor = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=VDN_DEVICE,
            )

            action_tensor = torch.as_tensor(
                actions,
                dtype=torch.long,
                device=VDN_DEVICE,
            )

            reward_tensor = torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=VDN_DEVICE,
            )

            next_obs_tensor = torch.as_tensor(
                next_observations,
                dtype=torch.float32,
                device=VDN_DEVICE,
            )

            done_tensor = torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=VDN_DEVICE,
            )

            weight_tensor = torch.as_tensor(
                importance_weights,
                dtype=torch.float32,
                device=VDN_DEVICE,
            )

            current_q = self.q(
                obs_tensor.reshape(
                    batch_count * agent_count,
                    self.d_in,
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
                next_q_online = self.q(
                    next_obs_tensor.reshape(
                        batch_count * agent_count,
                        self.d_in,
                    )
                ).reshape(
                    batch_count,
                    agent_count,
                    2,
                )

                next_q_target = self.q_tgt(
                    next_obs_tensor.reshape(
                        batch_count * agent_count,
                        self.d_in,
                    )
                ).reshape(
                    batch_count,
                    agent_count,
                    2,
                )

                if self.double_dqn:
                    ranking_q = next_q_online
                else:
                    ranking_q = next_q_target

                next_advantages = (
                    ranking_q[:, :, 1]
                    - ranking_q[:, :, 0]
                )

                selected_next_indices = (
                    next_advantages.topk(
                        min(
                            self.k_select,
                            agent_count,
                        ),
                        dim=1,
                    ).indices
                )

                next_actions = torch.zeros(
                    (batch_count, agent_count),
                    dtype=torch.long,
                    device=VDN_DEVICE,
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
                    * self.gamma
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
                weight_tensor * element_losses
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "A perda do VDN não é finita."
                )

            self.opt.zero_grad()
            loss.backward()

            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.q.parameters(),
                    self.grad_clip,
                )

            self.opt.step()

            self.buf.update_priorities(
                indices=indices,
                td_error_absolute=(
                    td_error_absolute
                ),
            )

            losses.append(
                float(loss.item())
            )

            td_means.append(
                float(td_error_absolute.mean())
            )

            self.total_updates += 1

        self._train_calls += 1

        if (
            self._train_calls
            % self.target_sync_every
            == 0
        ):
            self.q_tgt.load_state_dict(
                self.q.state_dict()
            )
            self.q_tgt.eval()
            self.last_target_sync = True

        self.last_td_abs_mean = float(
            np.mean(td_means)
        )

        return float(np.mean(losses))




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

        # Metadados de ataque são usados somente para auditoria e logs.
        # Eles não entram no estado observado pelo VDN.
        self.is_attacker_by_client = np.zeros(
            self.n_clients,
            dtype=bool,
        )
        self.attack_active_by_client = np.zeros(
            self.n_clients,
            dtype=bool,
        )
        self.selection_count_by_client = np.zeros(
            self.n_clients,
            dtype=np.int64,
        )
        self.active_attack_selection_count = 0


        # Parâmetros do experimento FEDMARL original.
        # Todos podem ser sobrescritos por variáveis de ambiente,
        # permitindo smoke tests sem editar novamente o Python.
        self.vdn_hidden = int(
            os.environ.get("VDN_HIDDEN", "128")
        )
        self.vdn_lr = float(
            os.environ.get("VDN_LR", "0.001")
        )
        self.vdn_weight_decay = float(
            os.environ.get("VDN_WEIGHT_DECAY", "0.0001")
        )
        self.vdn_gamma = float(
            os.environ.get("VDN_GAMMA", "0.9")
        )
        self.vdn_grad_clip = float(
            os.environ.get("VDN_GRAD_CLIP", "1.0")
        )
        self.vdn_target_sync_every = int(
            os.environ.get("VDN_TARGET_SYNC_EVERY", "20")
        )
        self.vdn_buffer_size = int(
            os.environ.get("VDN_BUFFER_SIZE", "20000")
        )

        self.reward_window = int(
            os.environ.get("REWARD_WINDOW", "5")
        )
        self.warmup_transitions = int(
            os.environ.get("VDN_WARMUP_TRANSITIONS", "50")
        )
        self.start_train_round = int(
            os.environ.get("VDN_START_TRAIN_ROUND", "50")
        )
        self.updates_per_round = int(
            os.environ.get("VDN_UPDATES_PER_ROUND", "50")
        )
        self.train_every = int(
            os.environ.get("VDN_TRAIN_EVERY", "1")
        )

        self.batch_base = int(
            os.environ.get("VDN_BATCH_BASE", "32")
        )
        self.batch_max = int(
            os.environ.get("VDN_BATCH_MAX", "256")
        )
        self.batch_buffer_ratio = int(
            os.environ.get("VDN_BATCH_BUFFER_RATIO", "4")
        )

        self.vdn_epsilon = float(
            os.environ.get("VDN_EPSILON", "0.15")
        )
        self.vdn_swap_m = int(
            os.environ.get("VDN_SWAP_M", "2")
        )

        self.per_alpha = float(
            os.environ.get("PER_ALPHA", "0.6")
        )
        self.per_beta_start = float(
            os.environ.get("PER_BETA_START", "0.4")
        )
        self.per_beta_end = float(
            os.environ.get("PER_BETA_END", "1.0")
        )
        self.per_beta_steps = int(
            os.environ.get("PER_BETA_STEPS", "4000")
        )
        self.per_eps = float(
            os.environ.get("PER_EPS", "0.001")
        )

        self.client_wait_timeout = float(
            os.environ.get("CLIENT_WAIT_TIMEOUT", "1800")
        )
        self.get_properties_timeout = float(
            os.environ.get("GET_PROPERTIES_TIMEOUT", "120")
        )

        if self.k_select <= 0 or self.k_select > self.n_clients:
            raise ValueError(
                "k_select deve estar entre 1 e n_clients."
            )
        if self.warmup_transitions < 0:
            raise ValueError(
                "VDN_WARMUP_TRANSITIONS não pode ser negativo."
            )
        if self.start_train_round <= 0:
            raise ValueError(
                "VDN_START_TRAIN_ROUND deve ser positivo."
            )
        if self.updates_per_round <= 0:
            raise ValueError(
                "VDN_UPDATES_PER_ROUND deve ser positivo."
            )
        if self.batch_base <= 0 or self.batch_max < self.batch_base:
            raise ValueError(
                "Configuração de batch do VDN inválida."
            )
        if not 0.0 <= self.vdn_epsilon <= 1.0:
            raise ValueError(
                "VDN_EPSILON deve estar entre 0 e 1."
            )

        self.vdn_selector = VDNSelector(
            n_agents=self.n_clients,
            d_in=5,
            k_select=self.k_select,
            hidden=self.vdn_hidden,
            lr=self.vdn_lr,
            weight_decay=self.vdn_weight_decay,
            gamma=self.vdn_gamma,
            grad_clip=self.vdn_grad_clip,
            target_sync_every=self.vdn_target_sync_every,
            buf_size=self.vdn_buffer_size,
            batch_size=self.batch_base,
            train_steps=self.updates_per_round,
            per_alpha=self.per_alpha,
            per_beta_start=self.per_beta_start,
            per_beta_end=self.per_beta_end,
            per_beta_steps=self.per_beta_steps,
            per_eps=self.per_eps,
            double_dqn=True,
            seed=self.selection_seed,
        )

        self.validation_loss_history = []
        self.pending_transition = None

        self.run_id = os.environ.get(
            "RUN_ID",
            "fedmarl_final",
        ).strip()
        if not self.run_id:
            raise ValueError("RUN_ID não pode ser vazio.")

        results_root = os.environ.get(
            "RESULTS_DIR",
            "/results",
        )
        self.results_directory = os.path.join(
            results_root,
            self.run_id,
        )
        os.makedirs(
            self.results_directory,
            exist_ok=True,
        )
        self.round_results_path = os.path.join(
            self.results_directory,
            "rounds.jsonl",
        )
        self.config_results_path = os.path.join(
            self.results_directory,
            "config.json",
        )

        config_snapshot = {
            "run_id": self.run_id,
            "n_clients": self.n_clients,
            "k_select": self.k_select,
            "total_rounds": self.total_rounds,
            "selection_seed": self.selection_seed,
            "warmup_transitions": self.warmup_transitions,
            "start_train_round": self.start_train_round,
            "updates_per_round": self.updates_per_round,
            "train_every": self.train_every,
            "batch_base": self.batch_base,
            "batch_max": self.batch_max,
            "batch_buffer_ratio": self.batch_buffer_ratio,
            "epsilon": self.vdn_epsilon,
            "swap_m": self.vdn_swap_m,
            "target_sync_every": self.vdn_target_sync_every,
            "buffer_size": self.vdn_buffer_size,
            "reward_window": self.reward_window,
        }
        with open(
            self.config_results_path,
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                config_snapshot,
                config_file,
                indent=2,
                sort_keys=True,
            )

        print(
            "[VDN configuração] "
            f"warmup={self.warmup_transitions}, "
            f"inicio_treino={self.start_train_round}, "
            f"updates_por_rodada={self.updates_per_round}, "
            f"batch={self.batch_base}-{self.batch_max}, "
            f"target_sync={self.vdn_target_sync_every}, "
            f"epsilon={self.vdn_epsilon}, "
            f"swap_m={self.vdn_swap_m}, "
            f"resultados={self.round_results_path}",
            flush=True,
        )




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
            timeout=self.client_wait_timeout,
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
                timeout=self.get_properties_timeout,
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

            is_attacker = bool(
                int(fit_result.metrics.get("is_attacker", 0))
            )
            attack_active = bool(
                int(fit_result.metrics.get("attack_active", 0))
            )
            flip_rate = float(
                fit_result.metrics.get("flip_rate", 0.0)
            )
            flipped_examples = int(
                fit_result.metrics.get("flipped_examples", 0)
            )

            if attack_active and not is_attacker:
                raise RuntimeError(
                    f"O cliente {registered_client_id} informou ataque ativo, "
                    "mas não está marcado como atacante."
                )

            if not 0.0 <= flip_rate <= 1.0:
                raise RuntimeError(
                    f"O cliente {registered_client_id} retornou "
                    f"flip_rate inválido: {flip_rate}."
                )

            self.is_attacker_by_client[
                int(registered_client_id)
            ] = is_attacker
            self.attack_active_by_client[
                int(registered_client_id)
            ] = attack_active

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
                f"projection_mom={projection_mom:+.6f}, "
                f"papel={'ATACANTE' if is_attacker else 'HONESTO'}, "
                f"ataque_ativo={int(attack_active)}, "
                f"flip_rate={flip_rate:.3f}, "
                f"rotulos_invertidos={flipped_examples}",
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

        force_random = (
            self.vdn_selector.buf.n
            < self.warmup_transitions
        )

        selection_mode = os.environ.get(
            "SELECTION_MODE",
            "vdn",
        ).strip().lower()

        selection_seed = int(
            os.environ.get(
                "RANDOM_SELECTION_SEED", 
                "4096",
            )
        )

        if selection_mode not in {"vdn", "random"}:
            raise ValueError(
                f"SELECTION_MODE inválido: "
                f"{selection_mode}"
            )

        if selection_mode == "random":

            candidate_ids = sorted(
                int(candidate[0])
                for candidate in candidate_results
            )

            if len(candidate_ids) < self.k_select:
                raise RuntimeError(
                    f"Há somente {len(candidate_ids)} "
                    f"atualizações válidas, mas "
                    f"{self.k_select} clientes devem "
                    f"ser selecionados."
                )

            rng = np.random.default_rng(
                selection_seed + server_round
            )

            selected_ranked_ids = [
                int(client_id)
                for client_id in rng.choice(
                    candidate_ids,
                    size=self.k_select,
                    replace=False,
                )
            ]

            actions = np.zeros(
                self.n_clients,
                dtype=np.int64,
            )

            actions[selected_ranked_ids] = 1

            # Mantém as variáveis esperadas pelo restante
            # do código, mas elas não participam da seleção.
            q_values = np.zeros(
                (self.n_clients, 2),
                dtype=np.float32,
            )

            advantages = np.zeros(
                self.n_clients,
                dtype=np.float32,
            )

            print(
                f"[RANDOM política] "
                f"seed={selection_seed}, "
                f"K={self.k_select}",
                flush=True,
            )

            print(
                f"[Rodada {server_round}] "
                f"seleção RANDOM="
                f"{selected_ranked_ids}",
                flush=True,
            )

        else:

            print(
                f"[VDN política] "
                f"modo="
                f"{'warmup-aleatório' if force_random else 'Top-K'}, "
                f"buffer_n={self.vdn_selector.buf.n}, "
                f"warmup={self.warmup_transitions}, "
                f"epsilon={self.vdn_epsilon:.3f}, "
                f"swap_m={self.vdn_swap_m}",
                flush=True,
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
                force_random=force_random,
            )

            for client_id in range(self.n_clients):
                print(
                    f"[VDN] CLIENT_ID={client_id}, "
                    f"Q0={q_values[client_id, 0]:+.6f}, "
                    f"Q1={q_values[client_id, 1]:+.6f}, "
                    f"advantage="
                    f"{advantages[client_id]:+.6f}, "
                    f"action="
                    f"{int(actions[client_id])}",
                    flush=True,
                )

            print(
                f"[Rodada {server_round}] "
                f"ranking Top-K do VDN="
                f"{selected_ranked_ids}",
                flush=True,
            )

        if int(actions.sum()) != self.k_select:
            raise RuntimeError(
                f"A política {selection_mode} selecionou "
                f"{int(actions.sum())} clientes, "
                f"mas deveria selecionar "
                f"{self.k_select}."
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

        self.selection_count_by_client[selected_ids] += 1

        selected_attacker_ids = [
            client_id
            for client_id in selected_ids
            if self.is_attacker_by_client[client_id]
        ]
        selected_active_attacker_ids = [
            client_id
            for client_id in selected_ids
            if self.attack_active_by_client[client_id]
        ]
        selected_honest_ids = [
            client_id
            for client_id in selected_ids
            if not self.is_attacker_by_client[client_id]
        ]

        self.active_attack_selection_count += len(
            selected_active_attacker_ids
        )

        print(
            f"[Auditoria seleção] round={server_round}, "
            f"honestos={selected_honest_ids}, "
            f"atacantes={selected_attacker_ids}, "
            f"atacantes_ativos={selected_active_attacker_ids}",
            flush=True,
        )

        attacker_ids = np.flatnonzero(
            self.is_attacker_by_client
        ).astype(int).tolist()
        honest_ids = np.flatnonzero(
            ~self.is_attacker_by_client
        ).astype(int).tolist()

        attacker_total_selections = int(
            self.selection_count_by_client[attacker_ids].sum()
        ) if attacker_ids else 0
        honest_total_selections = int(
            self.selection_count_by_client[honest_ids].sum()
        ) if honest_ids else 0

        print(
            f"[Auditoria acumulada] contagens_por_cliente="
            f"{self.selection_count_by_client.tolist()}, "
            f"selecoes_honestos={honest_total_selections}, "
            f"selecoes_atacantes={attacker_total_selections}, "
            f"selecoes_atacantes_ativos="
            f"{self.active_attack_selection_count}",
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


        
        # Média simples dos modelos selecionados.
        # Cada cliente recebe peso 1/K, como no experimento FEDMARL original.
        selected_model_ndarrays = [
            parameters_to_ndarrays(
                fit_result.parameters
            )
            for (
                _,
                _client_proxy,
                fit_result,
                _client_delta,
                _projection_mom,
            ) in selected_candidates
        ]

        if not selected_model_ndarrays:
            raise RuntimeError(
                "Nenhum modelo foi selecionado para agregação."
            )

        expected_layer_count = len(global_ndarrays)

        for position, client_ndarrays in enumerate(
            selected_model_ndarrays
        ):
            if len(client_ndarrays) != expected_layer_count:
                raise RuntimeError(
                    f"O modelo selecionado na posição {position} possui "
                    f"{len(client_ndarrays)} tensores, mas eram esperados "
                    f"{expected_layer_count}."
                )

        aggregated_ndarrays = []

        for layer_index, global_layer in enumerate(
            global_ndarrays
        ):
            client_layers = [
                np.asarray(
                    client_ndarrays[layer_index]
                )
                for client_ndarrays
                in selected_model_ndarrays
            ]

            expected_shape = np.asarray(
                global_layer
            ).shape

            for position, client_layer in enumerate(
                client_layers
            ):
                if client_layer.shape != expected_shape:
                    raise RuntimeError(
                        f"O tensor {layer_index} do modelo selecionado "
                        f"na posição {position} possui dimensão "
                        f"{client_layer.shape}, mas era esperada "
                        f"{expected_shape}."
                    )

            if np.issubdtype(
                np.asarray(global_layer).dtype,
                np.floating,
            ):
                aggregated_layer = np.mean(
                    np.stack(
                        client_layers,
                        axis=0,
                    ),
                    axis=0,
                    dtype=np.float64,
                ).astype(
                    np.asarray(global_layer).dtype,
                    copy=False,
                )
            else:
                first_layer = client_layers[0]

                if not all(
                    np.array_equal(
                        first_layer,
                        client_layer,
                    )
                    for client_layer
                    in client_layers[1:]
                ):
                    raise RuntimeError(
                        f"O tensor inteiro {layer_index} difere entre "
                        "os clientes selecionados."
                    )

                aggregated_layer = first_layer.copy()

            aggregated_ndarrays.append(
                aggregated_layer
            )

        aggregated_parameters = ndarrays_to_parameters(
            aggregated_ndarrays
        )

        selected_train_losses = [
            float(fit_result.metrics["metric"])
            for (
                _,
                _client_proxy,
                fit_result,
                _client_delta,
                _projection_mom,
            ) in selected_candidates
            if "metric" in fit_result.metrics
        ]

        metrics = {}

        if selected_train_losses:
            metrics["metric"] = float(
                np.mean(selected_train_losses)
            )

        equal_weight = 1.0 / len(
            selected_model_ndarrays
        )

        print(
            f"[Agregação] modo=média simples, "
            f"clientes={selected_ids}, "
            f"peso_por_cliente={equal_weight:.6f}",
            flush=True,
        )

        if metrics:
            print(
                f"Métrica agregada simples: {metrics}",
                flush=True,
            )

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(
                aggregated_parameters
            )

            net = set_weights(
                net,
                ndarrays,
            )
            

            (
                 validation_loss,
                validation_accuracy,
            ) = evaluate_server_metrics(
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
                f"val_accuracy="
                f"{100.0 * validation_accuracy:.2f}%, "               
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

            vdn_loss = None
            training_batch_size = None

            can_train = (
                server_round >= self.start_train_round
                and server_round % self.train_every == 0
                and self.vdn_selector.buf.n
                >= self.batch_base
            )

            if can_train and not force_random:
                training_batch_size = (
                    dynamic_batch_size(
                        buffer_size=(
                            self.vdn_selector.buf.n
                        ),
                        base=self.batch_base,
                        maximum=self.batch_max,
                        ratio=(
                            self.batch_buffer_ratio
                        ),
                    )
                )

                vdn_loss = self.vdn_selector.train(
                    batch_size=training_batch_size,
                    train_steps=(
                        self.updates_per_round
                    ),
                )

                print(
                    f"[VDN treino] "
                    f"round={server_round}, "
                    f"loss={vdn_loss:.8f}, "
                    f"batch={training_batch_size}, "
                    f"steps={self.updates_per_round}, "
                    f"beta="
                    f"{self.vdn_selector.last_train_beta:.6f}, "
                    f"td_abs_mean="
                    f"{self.vdn_selector.last_td_abs_mean:.6f}, "
                    f"train_calls="
                    f"{self.vdn_selector._train_calls}, "
                    f"updates="
                    f"{self.vdn_selector.total_updates}, "
                    f"target_sync="
                    f"{int(self.vdn_selector.last_target_sync)}",
                    flush=True,
                )

            else:
                print(
                    f"[VDN treino] aguardando, "
                    f"round={server_round}/"
                    f"{self.start_train_round}, "
                    f"buffer_n="
                    f"{self.vdn_selector.buf.n}/"
                    f"{self.batch_base}, "
                    f"force_random="
                    f"{int(force_random)}",
                    flush=True,
                )

            round_record = {
                "round": int(server_round),
                "validation_loss": float(validation_loss),
                "reward": float(reward),
                "selected_clients": [
                    int(client_id)
                    for client_id in selected_ids
                ],
                "selected_honest_clients": [
                    int(client_id)
                    for client_id in selected_honest_ids
                ],
                "selected_attacker_clients": [
                    int(client_id)
                    for client_id in selected_attacker_ids
                ],
                "selected_active_attacker_clients": [
                    int(client_id)
                    for client_id in selected_active_attacker_ids
                ],
                "aggregated_train_loss": (
                    float(metrics["metric"])
                    if "metric" in metrics
                    else None
                ),
                "vdn_loss": (
                    float(vdn_loss)
                    if vdn_loss is not None
                    else None
                ),
                "vdn_batch_size": (
                    int(training_batch_size)
                    if training_batch_size is not None
                    else None
                ),
                "replay_size": int(
                    self.vdn_selector.buf.n
                ),
                "vdn_total_updates": int(
                    self.vdn_selector.total_updates
                ),
                "probe_loss_by_client": [
                    float(value)
                    for value in probe_by_client.tolist()
                ],
                "projection_mom_by_client": [
                    float(value)
                    for value in projection_by_client.tolist()
                ],
                "advantage_by_client": [
                    float(value)
                    for value in advantages.tolist()
                ],
                "selection_count_by_client": [
                    int(value)
                    for value in self.selection_count_by_client.tolist()
                ],
            }

            with open(
                self.round_results_path,
                "a",
                encoding="utf-8",
            ) as results_file:
                json.dump(
                    round_record,
                    results_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                results_file.write("\n")
                results_file.flush()
                os.fsync(results_file.fileno())

            print(
                f"[Resultados] rodada {server_round} salva em "
                f"{self.round_results_path}",
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

        num_clients = int(
            os.environ.get(
                "NUM_CLIENTS",
                "50",
            )
        )

        num_rounds = int(
            os.environ.get(
                "NUM_ROUNDS",
                "500",
            )
        )


        select_clients = int(
            os.environ.get(
                "SELECT_CLIENTS",
                "15",
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
