import gym
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


@dataclass
class SARSd:
    state: object
    action: int
    reward: float
    next_state: object
    done: bool


class Memory:
    def __init__(self):
        self.size = 10
        self.deque = []

    def sample(self):
        return random.choice(self.deque)

    def record(self, item):
        self.deque.append(item)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(8, 32),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(32, 32),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(32, 4))

    def forward(self, x):
        output = self.regressor(x)
        return output


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.1
        self.batch_size = 64
        self.gamma = 0.99
        self.memory = Memory()
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.mse = nn.MSELoss()

    def Return_Q(self, state):
        return self.target_net.forward(state)

    def Copy_model(self):
        tau = 0.001
        params = zip(self.policy_net.parameters(), self.target_net.parameters())
        for pol, tar in params:
            tar.data.copy_(tau * pol.data + (1 - tau) * tar.data)

    def train(self):
        batch = [self.memory.sample() for i in range(self.batch_size)]
        states = [i.state for i in batch]
        actions = [i.action for i in batch]
        rewards = [i.reward for i in batch]
        dones = [i.done for i in batch]
        next_states = [i.next_state for i in batch]

        states = torch.stack(list(map(torch.tensor, states))).to(self.device)
        next_states = torch.stack(list(map(torch.tensor, next_states))).to(self.device)

        with torch.no_grad():
            prim_q = self.policy_net(next_states)
            targ_q = self.target_net(next_states)

        q_bests = torch.tensor([reward if done else
                               reward + self.gamma * tq[torch.argmax(pq).item()]
                               for reward, pq, done, tq in zip(rewards, prim_q, dones, targ_q)]).to(self.device)

        cur_q = self.policy_net(states)
        end_q = torch.stack([x[y] for x, y in zip(cur_q, actions)]).to(self.device)
        loss = self.mse(q_bests.float(), end_q.float())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # targets = []
        # for t in batch:
        #     next_state = t.next_state
        #     result = self.policy_net.forward(next_state)
        #     max_value = max(result)
        #     max_index = result.argmax()
        #     if not t.done:
        #         q = self.Return_Q(next_state)[max_index]
        #     else:
        #         q = torch.tensor(0)
        #     target = t.reward + self.gamma * q
        #     targets.append([next_state, target, t.action])
        # for i in targets:
        #     pred = self.policy_net(i[0])[i[2]]
        #     loss = self.mse(i[1], pred)
        #     self.optimiser.zero_grad()
        #     loss.backward()
        #     self.optimiser.step()
        #
        # # self.policy_net = self.target_net
        # pass

    def Set_weight(self):
        pass

    def Load_weight(self):
        pass


env = gym.make('LunarLander-v2')
agent = Agent()
df = pd.DataFrame(columns=['steps', 'reward'])

episodes = 5000
epsilon = 0.1
for i_episode in range(episodes):
    observation = env.reset()
    # 0 = nothing
    # 1 = right engine
    # 2 = bottom engine
    # 3 = left engine
    # print(env.observation_space)
    tot_reward = 0
    for t in range(200):
        if i_episode >= episodes-21:
            env.render()
        # observation = [x, y, x_vel, y_vel, lander_angle, angle_vel, left_contact, right_contact]
        # print(observation)
        old_observation = observation.copy()

        max_index = torch.argmax(agent.policy_net(torch.from_numpy(observation))).item()
        # max_index = agent.policy_net(observation).argmax()
        weights = [epsilon / 4 for i in range(4)]
        weights[max_index] = 1 - epsilon + epsilon/4
        action = random.choices(population=[0, 1, 2, 3], weights=weights)[0]
        observation, reward, done, info = env.step(action)

        agent.memory.record(SARSd(state=old_observation, action=action, reward=reward, next_state=observation, done=done))
        tot_reward += reward
        if len(agent.memory.deque) >= agent.batch_size:
            if t % 4 == 0:
                agent.train()
                agent.Copy_model()
        if done:
            df = df.append({'steps': i_episode, 'reward': tot_reward}, ignore_index=True)
            print(f"Episode {i_episode} finished after {t + 1} timesteps, average reward is {tot_reward / (t + 1)}, total is {tot_reward}")
            break
    env.close()

df['reward'].plot()
plt.show()

