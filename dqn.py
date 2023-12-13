import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn

EPISODES = 300



class NN(nn.Module):
    def __init__(self, input_size, output_size):

        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size,render):

        self.render = render
        self.load_model = True

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 200

        self.memory = deque(maxlen=2000)

        self.model = NN(state_size,action_size)
        self.target_model = NN(state_size,action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


        self.update_target_model()

        if self.load_model:
            self.model.load_state_dict(torch.load('./weights3.pt'))
            self.epsilon = self.epsilon_min



    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())



    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(torch.from_numpy(state).float()).detach().numpy()
            return np.argmax(q_value[0])


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model(torch.from_numpy(update_input).float()).detach().numpy()
        target_val = self.target_model(torch.from_numpy(update_target).float()).detach().numpy()

        for i in range(self.batch_size):

            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))


        update_input = torch.from_numpy(update_input).float()
        target = torch.from_numpy(target).float()
        scores = self.model(update_input)
        loss = self.criterion(scores, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size,True)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()


            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward = reward if not done or score == 499 else -100


            agent.append_sample(state, action, reward, next_state, done)

            agent.train_model()
            score += reward
            state = next_state

            if done:

                agent.update_target_model()


                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)



                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()


        if e % 50 == 0:
            torch.save(agent.model.state_dict(), './weights.pt')
