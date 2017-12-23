#coding=utf-8
from Blackjack import Blackjack
import numpy as np
from collections import defaultdict
import plotting
import sys

def epsilon_greedy_policy(Q,observation,nA,epsilon):
    best_action = np.argmax(Q[observation])
    A = np.ones(nA,dtype=np.float32)*epsilon/nA
    A[best_action] += 1-epsilon
    return A

def MC_Control_with_epsilon_greedy(env,episode_nums,discount_factor=1.0, epsilon=0.1):

    env = Blackjack()
    Q = defaultdict(lambda:np.zeros(env.nA))
    return_sum=defaultdict(float)
    return_count=defaultdict(float)

    for i_episode in range(1,1+episode_nums):
        env._reset()
        state = env.observation()
        episode=[]
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, episode_nums))
            sys.stdout.flush()
        for i in range(100):

            A = epsilon_greedy_policy(Q,state,env.nA,epsilon)

            probs = A
            action = np.random.choice(np.arange(env.nA),p=probs)

            next_state,reward,done = env._step(action)
            episode.append((state,action,reward))
            if done:
                break
            else:
                state = next_state

        seperate_episode = set([(tuple(x[0]), x[1]) for x in episode])

        for state,action in seperate_episode:
            for idx,e in enumerate(episode):
                if e[0]==state and e[1]==action:
                    first_visit_idx = idx
                    break
            pair = (state,action)
            G = sum([e[2]*(discount_factor**i) for i,e in enumerate(episode[first_visit_idx:])])
            return_sum[pair]+=G
            return_count[pair]+=1.0
            Q[state][action]=return_sum[pair]*1.0/return_count[pair]
    return Q

env=Blackjack()
Q = MC_Control_with_epsilon_greedy(env, episode_nums=500000, epsilon=0.1)

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")







