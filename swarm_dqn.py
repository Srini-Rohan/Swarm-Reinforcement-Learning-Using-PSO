import sys
import gym
import panda_gym
import pylab
import random
import numpy as np
from collections import deque
from dqn import NN,DQNAgent
import torch
import matplotlib.pyplot as plt


EPISODES = 300


class DQN_PSO_Agent:
    def __init__(self,state_size, action_size,num_agents):
        self.render = True
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=10000)
        self.agents = [DQNAgent(state_size, action_size,False) for i in range(num_agents)]
        self.start_exchange = 400
        self.E = [-float('inf') for i in range(num_agents)]
        self.E_p = [-float('inf') for i in range(num_agents)]
        self.E_G = -float('inf')
        self.presonal_best = [NN(state_size, action_size) for i in range(num_agents)]
        self.global_best = NN(state_size,action_size)

        self.num_episodes_exchange = 5
        self.batch_size = 128

        self.C1 = 2.2
        self.C2 = 2.2

    def update_best(self):
        for i in range(num_agents):
            if(self.E[i]>self.E_p[i]):
                self.E_p[i] = self.E[i]
                self.presonal_best[i].load_state_dict(self.agents[i].model.state_dict())
            if(self.E[i]>self.E_G):
                self.E_G = self.E[i]
                self.global_best.load_state_dict(self.agents[i].model.state_dict())

    def update_q(self):
        if len(self.memory) < self.start_exchange:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = [self.agents[i].model(torch.from_numpy(update_input).float()).detach().numpy() for i in range(num_agents)]
        presonal_best = [self.presonal_best[i](torch.from_numpy(update_input).float()).detach().numpy() for i in range(num_agents)]
        global_best = self.global_best(torch.from_numpy(update_input).float()).detach().numpy()


        for i in range(batch_size):
            for j in range(self.num_agents):

                    R1 = np.random.uniform(0,0.5)
                    R2 = np.random.uniform(0,0.5)


                    target[j][i][action[i]] += self.C1*R1*(presonal_best[j][i][action[i]]-target[j][i][action[i]]) + self.C2*R2*(global_best[i][action[i]]-target[j][i][action[i]])

        update_input = torch.from_numpy(update_input).float()
        for i in range(self.state_size):
            target[i] = torch.from_numpy(target[i]).float()
        for j in range(20):
            for i  in range(self.num_agents):
                if(self.E[i]>400):
                    continue
                scores = self.agents[i].model(update_input)
                loss = self.agents[i].criterion(scores, target[i])

                self.agents[i].optimizer.zero_grad()
                loss.backward()
                self.agents[i].optimizer.step()

if __name__ == "__main__":

    num_agents = 4

    env = [gym.make('CartPole-v1') for i in range(num_agents)]

    state_size = env[0].observation_space.shape[0]
    action_size = env[0].action_space.n

    multi_agent = DQN_PSO_Agent(state_size, action_size,4)

    scores, episodes = [[] for i in range(num_agents)], [[] for i in range(num_agents)]
    figure, axis = plt.subplots(4)

    for e in range(EPISODES):
        scores_indi = []
        for i in range(num_agents):
            done = False
            score = 0
            state = env[i].reset()
            state = np.reshape(state, [1, state_size])
            return_indi = 0
            step = 0
            while not done:
                step +=1
                env[i].render()

                action = multi_agent.agents[i].get_action(state)
                next_state, reward, done, info = env[i].step(action)
                next_state = np.reshape(next_state, [1, state_size])

                reward = reward if not done or score == 499 else -100


                multi_agent.agents[i].append_sample(state, action, reward, next_state, done)
                multi_agent.memory.append((state, action, reward, next_state, done))
                multi_agent.agents[i].train_model()
                score += reward
                return_indi += reward/(0.999)**step
                state = next_state

                if done:

                    multi_agent.agents[i].update_target_model()

                    score = score if score == 500 else score + 100
                    multi_agent.E[i] = score
                    scores[i].append(score)
                    episodes[i].append(e)
                    axis[i].plot(episodes[i], scores[i])
                    pylab.savefig("./dqn.png")
                    scores_indi.append(score)

                    if np.mean(scores[i][-min(10, len(scores[i])):]) > 475:
                        sys.exit()


            if e % 50 == 0:
                torch.save(multi_agent.agents[i].model.state_dict(), './weights'+str(i)+'.pt')
        multi_agent.update_best()
        if e>50:
            multi_agent.update_q()

        print("episode:", e, "  score:", scores_indi, "  memory length:",len(multi_agent.agents[0].memory), "  epsilon:", multi_agent.agents[0].epsilon)
