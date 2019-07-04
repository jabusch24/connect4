import numpy as np
import matplotlib.pyplot as plt
from game import Game
import os
import time
from util.getkey import get_manual_arrow_key


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# define actions
actions = (1,2,3,4,5,6,7)

# define Q-Network
class QNetwork(nn.Module):

    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        # TODO YOUR CODE HERE - simple network
        self.fc1 = torch.nn.Linear(in_features=state_space, out_features=state_space)
        self.fc2 = torch.nn.Linear(in_features=state_space, out_features=state_space)
        self.fc3 = torch.nn.Linear(in_features=state_space, out_features=action_space)
        self.relu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=0)
        self.state_space = state_space

    def forward(self, x):
        #x = self.one_hot_encoding(x)
        # TODO YOUR CODE HERE
        x = torch.from_numpy(x)
        x = x.to('cpu', torch.float)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    def one_hot_encoding(self, x):
        '''
        One-hot encodes the input data, based on the defined state_space.
        '''
        out_tensor = torch.zeros([1, state_space])
        out_tensor[0][x] = 1
        return out_tensor

def feedforward(model):
    # Choose an action by greedily (with e chance of random action) from the Q-network
    with torch.no_grad():
        # TODO YOUR CODE HERE
        # Do a feedforward pass for the current state s to get predicted Q-values
        # for all actions and use the max as action a: max_a Q(s, a)
        a = torch.argmax(model(s.flatten())).item()


    # e greedy exploration
    if np.random.rand(1) < e:
        a = np.random.randint(1, 7)

    # Get new state and reward from environment
    # TODO YOUR CODE HERE
    r, s1, game_over = game.perform_action(actions[a])
    # time.sleep(0.5)
    return r, s1, game_over

def backpropagate(model, criterion, optimizer, r, s, s1, game_over, r_list):

    # perform action to get reward r, next state s1 and game_over flag
    # calculate maximum overall network outputs: max_a’ Q(s1, a’).
    a1 = model(s1.flatten())

    # Calculate Q and target Q
    q = model(s.flatten()).max(0)[0].view(1, 1)
    # print("q", q, flush=True)

    with torch.no_grad():
        # Set target Q-value for action to: r + y max_a’ Q(s’, a’)
        target_q = r + y*torch.max(a1).view(1,1)
        # print("target_q", target_q, flush=True)

    # Calculate loss
    loss = criterion(q, target_q)
    if j == 1 and i % 100 == 0:
    # if r == 1 or r == -1:
        print("loss and reward: ", i, loss, r)
        print("q and q target: ", q, target_q)


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Add reward to list
    r_list += r

    # Replace old state with new
    s = s1

    return s, r_list, game_over

# Make use of cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Game Instance
game = Game(render=False)

# Define State and Action Space
state_space = game.max_row * game.max_col
action_space = len(actions)

# Set learning parameters
e = 0.5  # epsilon
lr = .1  # learning rate
y = .7  # discount factor
num_episodes = 10000

# create lists to contain total rewards and steps per episode
jList = []
rAList = []
rBList = []

# init Q-Network
agentA = QNetwork(state_space, action_space)
agentB = QNetwork(state_space, action_space)

# load existing model if exists:
if 'modelA.pt' in os.listdir('./'):
    agentA.load_state_dict(torch.load('./modelA.pt'))
    print("Model's state_dict:")
    for param_tensor in agentA.state_dict():
        print(param_tensor, "\t", agentA.state_dict()[param_tensor].size())

# load existing model if exists:
if 'modelB.pt' in os.listdir('./'):
    agentB.load_state_dict(torch.load('./modelB.pt'))
    print("Model's state_dict:")
    for param_tensor in agentB.state_dict():
        print(param_tensor, "\t", agentB.state_dict()[param_tensor].size())

# define optimizer and loss
# optimizer = optim.SGD(agent.parameters(), lr=lr)
optimizerA = optim.Adam(params=agentA.parameters())
optimizerB = optim.Adam(params=agentB.parameters())

# criterion = nn.SmoothL1Loss()
criterion = nn.SmoothL1Loss()

Awins = 0
Bwins = 0
for i in range(num_episodes):
    if i % 100 == 1:
        print('episode: ', i, flush=True)
        print("\Average steps per episode: " + str(sum(jList)/i), flush=True)
        print("\nScore over time A: " + str(sum(rAList)/i), flush=True)
        print("\nScore over time B: " + str(sum(rBList)/i), flush=True)
    # Reset environment and get first new observation
    s = game.reset()
    rA = 0
    rB = 0
    j = 0
    # The Q-Network learning algorithm
    while game.game_over == False:

        j += 1
        # time.sleep(0.5)
        if game.player == True:
            
            r, s1, game_over = feedforward(agentA)
            _, rA, _ = backpropagate(agentA, criterion, optimizerA, r[0], s, s1, game_over, rA)
            s, rB, game_over = backpropagate(agentB, criterion, optimizerB, r[1], s, s1, game_over, rB)
            if r[0] == 1:
                Awins += 1
        else:
            r, s1, game_over = feedforward(agentB)
            _, rB, _ = backpropagate(agentB, criterion, optimizerB, r[0], s, s1, game_over, rB)
            s, rA, game_over = backpropagate(agentA, criterion, optimizerA, r[1], s, s1, game_over, rA)
            if r[0] == 1:
                Bwins += 1

        if game_over:
            # Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    rAList.append(rA)
    rBList.append(rB)
    jList.append(j)

torch.save(agentA.state_dict(), './modelA.pt')
torch.save(agentB.state_dict(), './modelB.pt')
print("Awins:", Awins)
print("Bwins:", Bwins)

print("\Average steps per episode: " + str(sum(jList)/num_episodes))
print("\nScore over time A: " + str(sum(rAList)/num_episodes))
print("\nScore over time B: " + str(sum(rBList)/num_episodes))
# plt.plot(jList)
# plt.savefig("j_q_network.png")
# plt.show()
# plt.plot(rList)
# plt.show()
