#coding=utf-8

from Blackjack import Blackjack
from collections import defaultdict
import sys
import matplotlib
import os

import plotting

matplotlib.style.use('ggplot')


def sample_policy(observation):

    player_sum,dealer_show_hand,usable_ace=observation
    if player_sum<20:
        return 1
    return 0

#
# episode.append(state)
#                 if return_sum.has_key(state):
#                     return_sum[state]+=reward
#                 else:
#                     return_sum[state]=reward
#                 if return_count.has_key(state):
#                     return_count[state]+=1
#                 else:
#                     return_count=1
#                 if V.has_key(state):
#                     V[state]
def mc_prediction(policy,env,num_episodes,discount=1.0):
    """
    first-visit mc prediction
    :param env: blackjack env
    :param policy: initial policy
    :param num_episodes:
    :param discount:
    :return:
    """


    return_sum = defaultdict(float)
    return_count = defaultdict(float)
    V = defaultdict(float)

    for i_episode in range(1,1+num_episodes):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
            sys.stdout.flush()

        env._reset()
        state = env.observation()

        episode=[]

        #sample using given policy
        for i in range(100):
            action = policy(state)

            next_state,reward,done = env._step(action)
            episode.append((state,action,reward))
            if done:
                break
            else:
                state=next_state

        seperate_episode = set([tuple(eps[0]) for eps in episode])

        for s_eps in seperate_episode:
            #find the first visit state

            for i,x in enumerate(episode):
                if x[0] == s_eps:
                    first_visit_pos=i
            G = sum([e[2]*discount**idx for idx,e in enumerate(episode[first_visit_pos:])])

            return_sum[s_eps]+=G
            return_count[s_eps]+=1.0
            V[s_eps] = return_sum[s_eps]*1.0/return_count[s_eps]

    return V

env = Blackjack()

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
















