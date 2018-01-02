#coding=utf-8
import numpy as np
from collections import defaultdict
from CliffEnvironment import CliffEnvironment
import sys
import matplotlib.pyplot as plt

def epsilon_greedy_policy(Q,state,nA,epsilon=0.1):
    best_action = np.argmax(Q[state])
    A =np.ones(nA,dtype=np.float32)*epsilon/nA
    A[best_action] += 1-epsilon
    return A

def greedy_policy(Q,state):
    best_action = np.argmax(Q[state])
    return best_action


def plot(x,y):
    size = len(x)
    x = [x[i] for i in range(size) if i%10==0 ]
    y = [y[i] for i in range(size) if i%10==0 ]
    plt.plot(x, y, 'ro-')
    plt.ylim(-300, 0)
    plt.show()

def print_policy(Q,env):
    env = CliffEnvironment()
    result=""
    for i in range(env.height):
        line=""
        for j in range(env.width):
            action = np.argmax(Q[(j,i)])
            if action==0:
                line+="up    "
            elif action==1:
                line+="down  "
            elif action==2:
                line+="left  "
            else:
                line+="right "
        result=line+"\n"+result
    print result



def Q_learning(env,episode_nums,discount_factor=1.0, alpha=0.5,epsilon=0.1):

    env = CliffEnvironment()
    Q = defaultdict(lambda:np.zeros(env.nA))
    rewards=[]

    for i_episode in range(1,1+episode_nums):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, episode_nums))
            sys.stdout.flush()

        env._reset()
        state,done = env.observation()

        sum_reward=0.0

        while not done:

            A= epsilon_greedy_policy(Q,state,env.nA)
            probs = A
            action = np.random.choice(np.arange(env.nA),p=probs)
            new_state,reward,done = env._step(action)



            if done:
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor*0.0-Q[state][action])
                break
            else:
                new_action = greedy_policy(Q,new_state)
                Q[state][action]=Q[state][action]+alpha*(reward+discount_factor*Q[new_state][new_action]-Q[state][action])
                state = new_state
            sum_reward+=reward
        rewards.append(sum_reward)
    return Q,rewards

env = CliffEnvironment()
episode_nums=1000
Q,rewards = Q_learning(env,episode_nums)
print_policy(Q,env)

average_rewards =[]
for i in range(10):
    _,rewards = Q_learning(env,episode_nums)
    if len(average_rewards)==0:
        average_rewards=np.array(rewards)
    else:
        average_rewards=average_rewards+np.array(rewards)
average_rewards=average_rewards*1.0/10
plot(range(1,1+episode_nums),average_rewards)



















