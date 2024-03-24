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
        # 这里使用了 torch.no_grad() 上下文管理器，确保在接下来的计算过程中不会追踪梯度，因
        # 为目标 Q 值是不需要进行梯度下降的。
        with torch.no_grad():  # target_Q has no gradient
            # 首先计算目标 Q 值 target_Q。
            # 使用目标 Actor 网络 self.actor_target 根据下一个状态 batch_s_ 计算出下一个状态的动作，
            # 然后输入到目标 Critic 网络 self.critic_target 中得到下一个状态的 Q 值 Q_。
            # 然后根据 Bellman 方程，
            # 将当前奖励 batch_r 与折扣因子 GAMMA 乘以下一个状态的 Q 值 Q_ 相加，
            # 乘以终止标志的衰减系数 (1 - batch_dw)，得到目标 Q 值 target_Q。
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_
        # 接着计算当前状态的 Q 值 current_Q，
        # 这是通过输入当前状态 batch_s 和动作 batch_a 到 Critic 网络 self.critic 中得到的。
        # 然后使用均方误差损失函数 F.mse_loss 计算 Critic 损失 critic_loss，
        # 用于衡量当前 Q 值和目标 Q 值之间的差异。
        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q, target_Q)
        # td_error = (target_Q - current_Q).detach().mean()
        # Optimize the critic
        # 接下来，对 Critic 网络进行优化。首先使用优化器的 zero_grad() 方法清零梯度，
        # 然后调用 backward() 方法进行反向传播计算梯度，
        # 最后调用 step() 方法更新网络参数。
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 接着，将 Critic 网络的参数设置为不需要计算梯度，以防止在 Actor 网络的训练中浪费计算资源。
        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        # 然后计算 Actor 损失 actor_loss，
        # 这是通过将当前状态 batch_s 输入到 Actor 网络 self.actor 中得到的动作，
        # 再将其输入到 Critic 网络 self.critic 中得到对应的 Q 值，然后取平均值并取负号。
        # 这个损失是希望最大化 Critic 网络对 Actor 输出动作的 Q 值。
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        # 接着对Actor网络进行优化。先使用优化器的zero_grad() 方法清零梯度，
        # 然后调用 backward() 方法进行反向传播计算梯度，
        # 最后调用 step() 方法更新网络参数。
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.actor_optimizer.step()
        # Unfreeze critic networks
        # 解冻 Critic 网络的参数，以便在下一次训练时能够更新 Critic 网络的参数。
        for params in self.critic.parameters():
            params.requires_grad = True
        # 最后，使用软更新（Soft Update）方法更新 Critic 目标网络的参数。
        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        # 同样地，使用软更新方法更新 Actor 目标网络的参数。
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        # for name, param in self.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f'Parameter: {name}, Gradient: {param.grad}')

        return actor_loss.data.numpy(), critic_loss.data.numpy()
