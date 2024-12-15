import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import Union, List, Dict, Tuple
from torch.distributions import Categorical


class A2CModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(state: torch.Tensor, compute_policy=True, compute_value=True,
                task_label=None):
        raise NotImplementedError()

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor, task_label=None):
        _, policy_logits = self(
            observation, compute_value=False, task_label=task_label)
        return Categorical(logits=policy_logits).sample()


class ActorCriticMLP(A2CModel):
    def __init__(
            self, num_inputs, num_actions,
            actor_hidden_sizes: Union[int, List[int]] = [64, 64],
            critic_hidden_sizes: Union[int, List[int]] = [64, 64],
            activation_type: str = 'relu'):
        super(ActorCriticMLP, self).__init__()
        # these are actually 2 models in one
        if type(actor_hidden_sizes) is int:
            actor_hidden_sizes = [actor_hidden_sizes]
        if type(critic_hidden_sizes) is int:
            critic_hidden_sizes = [critic_hidden_sizes]
        assert len(critic_hidden_sizes) and len(actor_hidden_sizes)
        if activation_type == 'relu':
            act = nn.ReLU()
        elif activation_type == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type {activation_type}")

        critic = [nn.Linear(
                      critic_hidden_sizes[i],
                      critic_hidden_sizes[i + 1])
                  for i in range(len(critic_hidden_sizes) - 1)]
        actor = [
            nn.Linear(actor_hidden_sizes[i],
                      actor_hidden_sizes[i + 1])
            for i in range(len(actor_hidden_sizes) - 1)]

        # self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.critic = []
        for layer in [nn.Linear(num_inputs, critic_hidden_sizes[0])]+critic:
            self.critic.append(layer)
            self.critic.append(act)
        self.critic.append(nn.Linear(critic_hidden_sizes[-1], num_actions))
        self.critic = nn.Sequential(*self.critic)

        self.actor = []
        for layer in [nn.Linear(num_inputs, actor_hidden_sizes[0])]+actor:
            self.actor.append(layer)
            self.actor.append(act)
        self.actor.append(nn.Linear(actor_hidden_sizes[-1], num_actions))
        self.actor = nn.Sequential(*self.actor)

    def forward(self, state: torch.Tensor, compute_policy=True,
                compute_value=True, task_label=None):
        value, policy_logits = None, None
        if compute_value:
            value = self.critic(state)
        if compute_policy:
            policy_logits = self.actor(state)

        return value, policy_logits

class MultiEnvActorCriticMLP(A2CModel):
    def __init__(
        self,
        task_id_to_model_shape: Dict[int, Tuple[int, int]],
        actor_hidden_sizes: Union[int, List[int]] = [64, 64],
        critic_hidden_sizes: Union[int, List[int]] = [64, 64],
        activation_type: str = 'relu'
    ):
        super(MultiEnvActorCriticMLP, self).__init__()

        if isinstance(actor_hidden_sizes, int):
            actor_hidden_sizes = [actor_hidden_sizes]
        if isinstance(critic_hidden_sizes, int):
            critic_hidden_sizes = [critic_hidden_sizes]

        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

        self.task_id_to_model_shape = task_id_to_model_shape

        # Shared layers
        self.shared_critic_hidden = nn.ModuleList(
            [nn.Linear(critic_hidden_sizes[i], critic_hidden_sizes[i + 1])
             for i in range(len(critic_hidden_sizes) - 1)]
        )
        self.shared_actor_hidden = nn.ModuleList(
            [nn.Linear(actor_hidden_sizes[i], actor_hidden_sizes[i + 1])
             for i in range(len(actor_hidden_sizes) - 1)]
        )

        # Task-specific layers stored in ModuleDict
        self.task_specific_critics = nn.ModuleDict()
        self.task_specific_actors = nn.ModuleDict()

        for task_id, (input_size, num_actions) in task_id_to_model_shape.items():
            self.task_specific_critics[task_id] = self._build_task_specific_layer(
                input_size, num_actions, self.shared_critic_hidden)
            self.task_specific_actors[task_id] = self._build_task_specific_layer(
                input_size, num_actions, self.shared_actor_hidden)

    def _build_task_specific_layer(self, input_size: int, num_actions: int, shared_hidden: nn.ModuleList):
        layers = [nn.Linear(input_size, shared_hidden[0].in_features), self.activation]
        layers += [layer for layer in shared_hidden]
        layers.append(nn.Linear(shared_hidden[-1].out_features, num_actions))
        return nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        task_label: str = None,
        compute_policy=True,
        compute_value=True
    ):
        if task_label not in self.task_id_to_model_shape: 
            raise ValueError(f"Task ID {task_label} not found in task_id_to_model_shape mapping")

        critic = self.task_specific_critics[task_label]
        actor = self.task_specific_actors[task_label]

        value, policy_logits = None, None
        if compute_value:
            value = critic(state)
        if compute_policy:
            policy_logits = actor(state)

        return value, policy_logits


class ConvActorCritic(A2CModel):
    """
        Smaller version of the Convolutional DQN network introduced in
        Mnih et al 2013 (DQN paper), re-used for experiments in
        Mnih et al. 2016 (A3C paper).
    """
    def __init__(self, input_channels, image_shape, n_actions,
                 batch_norm=False):
        super(ConvActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        # "We typically use a convolutional neural network
        # that has one softmax output for the policy π(at|st; θ) and
        # one linear output for the value function V (st; θv), with all
        # non-output layers shared."
        self.fc = nn.Sequential(
            nn.Linear(
                self._compute_flattened_shape(
                    (input_channels, image_shape[0],
                     image_shape[1])),
                256),
            nn.ReLU())

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, n_actions)

    def forward(self, x, compute_policy=True, compute_value=True,
                task_label=None):
        value, policy_logits = None, None

        # shared backbone of the actor-critic network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.fc(x.flatten(1))

        if compute_policy:
            # actor logits output head
            policy_logits = self.actor(x)
        if compute_value:
            # value output head
            value = self.critic(x)

        return value, policy_logits

    def _compute_flattened_shape(self, input_shape):
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
        return x.squeeze(0).flatten().shape[0]
