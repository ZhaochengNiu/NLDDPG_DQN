import torch
from model import Actor, Critic
import copy
import torch.nn.functional as F
import hyperparameter as hp
class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = hp.BATCH_SIZE  # batch size
        self.GAMMA = 0.9  # discount factor
        self.TAU = hp.TAU  # Softly update the target network
        self.lr = 0.01  # learning rate
        self.weight_decay = 0.01
        self.actor = Actor(state_dim, action_dim, self.hidden_width)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    # torch.tensor(s, dtype=torch.float)：将状态 s 转换为 PyTorch 的张量，并指定数据类型为浮点数。
    # torch.unsqueeze(..., 0)：在张量的第 0 维度上添加一个额外的维度，这是因为通常情况下，神经网络的输入需要是一个批次（batch），因此需要添加一个批次维度。
    # .data.numpy()：将预测结果转换为 NumPy 数组，这样可以更方便地进行后续处理。
    # .flatten()：将 NumPy 数组展平为一维数组，以便于后续处理。

    def learn(self, batch_s, batch_a, batch_r, batch_dw, batch_s_):

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q, target_Q)
        # td_error = (target_Q - current_Q).detach().mean()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()


        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        # for name, param in self.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f'Parameter: {name}, Gradient: {param.grad}')

        return actor_loss.data.numpy(), critic_loss.data.numpy()
