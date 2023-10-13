
import math
import gym
import numpy as np
#######################previous information###############################
env = gym.make('CartPole-v0')
#space of action and observation
print("env.action_space",env.action_space)
#The Discrete space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1

print("env.observation_space",env.observation_space)
#The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers

#check the box bound
print("env.observation_space.high",env.observation_space.high)
print("env.observation_space.low",env.observation_space.low)

#########################################################################
###################insert the seting##########################
#gym.spaces.Discrete(number_of_elements)
# x = gym.space.sample()
# assert gym.space.contains(x)
# assert gym.space.n == number_of_elements
#############################################
#discretize the spaces
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)
Maximum_reward=0

ALPHA = 0.1
GAMMA = 0.9
EPS = 1.0
numGames = 100000
totalRewards = np.zeros(numGames)
#####################statespace##########################
states = []
for i in range(len(cartPosSpace)+1):
    for j in range(len(cartVelSpace)+1):
        for k in range(len(poleThetaSpace)+1):
            for l in range(len(poleThetaVelSpace)+1):
                states.append((i,j,k,l))
#print(states)

####################Q-matrix#########################
Q1={}
for s in states:
    for a in range(2):
        Q1[s, a] = 0
#print("Q1",Q1)
##############################################

# def handmade_action(observation):
#     pos, v, ang, rot = observation
#     return 0 if ang < 0 else 1 # 柱子左傾則小車左移，否則右移 
####once the ang<0, move left ;otherwise, move right.

#epsilon greedy and Q-learning
def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon: # 有 ε 的機率會選擇隨機 action
        return action_space.sample() 
    else: # 其他時間根據現有 policy 選擇 action，也就是在 Q table 裡目前 state 中，選擇擁有最大 Q value 的 action
        return np.argmax(q_table[state]) 


def getState(observation):
    cartX, cartXdot, cartTheta, cartThetadot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetadot = int(np.digitize(cartThetadot, poleThetaVelSpace))

    return (cartX, cartXdot, cartTheta, cartThetadot)

def maxAction(Q1, state):    
    values = np.array([Q1[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

def plotRunningAverage(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
    return tuple(state)

for i in range(numGames):
    print('starting game ', i)
    #done = False
    epRewards = 0
    observation = env.reset()
    for j in range(500):
        s = getState(observation)
        #print(s)
        rand = np.random.random()
        a = maxAction(Q1,s) if rand < (1-EPS) else env.action_space.sample()
        observation_, reward, done, info = env.step(a)
        epRewards += reward
        #Q learning feature "think about future feebback"
        s_ = getState(observation_)
        a_ = maxAction(Q1,s_)
        Q1[s,a] = Q1[s,a] + ALPHA*(reward + GAMMA*Q1[s_,a_] - Q1[s,a])
        #print(Q1)
        observation = observation_
    EPS -= 2/(numGames) if EPS > 0 else 0
    totalRewards[i] = epRewards
    
    #plt.plot(totalRewards, 'b--')
    #plt.show()
    #plotRunningAverage(totalRewards)