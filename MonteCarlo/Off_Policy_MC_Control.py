#coding=utf-8
from Blackjack import Blackjack
import numpy as np
from collections import defaultdict
import plotting
import sys

def greedy_policy(Q,observation,nA):
    best_action = np.argmax(Q[observation])
    A = np.zeros(nA,dtype=np.float32)
    A[best_action] =1.0
    return A

def sample_policy(Q,observation,nA):
    A = np.ones(nA,dtype=np.float32)/nA
    return A

def Off_policy_MC_Control(env,episode_nums,discount_factor=1.0):

    env = Blackjack()
    Q = defaultdict(lambda:np.zeros(env.nA))
    target_policy = defaultdict(float)

    return_count=defaultdict(float)


    for i_episode in range(1,1+episode_nums):
        env._reset()
        state = env.observation()
        episode=[]
        prob_b=[]
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, episode_nums))
            sys.stdout.flush()
        for i in range(100):

            A = sample_policy(Q,state,env.nA)
            probs = A
            action = np.random.choice(np.arange(env.nA),p=probs)

            next_state,reward,done = env._step(action)
            episode.append((state,action,reward))
            prob_b.append(probs[action])
            if done:
                break
            else:
                state = next_state

        seperate_episode = set([(tuple(x[0]), x[1]) for x in episode])

        G =0.0
        W =1
        prob_b=prob_b[::-1]
        for idx,eps in enumerate(episode[::-1]):
            state,action,reward  = eps
            pair=(state,action)
            G = discount_factor*G+reward
            return_count[pair]+=W
            Q[state][action]+=W*1.0/return_count[pair]*(G-Q[state][action])
            target_policy[state] = np.argmax(Q[state])
            if target_policy[state]!=action:
                break
            W = W*1.0/prob_b[idx]

    return Q

env=Blackjack()
Q = Off_policy_MC_Control(env, episode_nums=500000)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")







