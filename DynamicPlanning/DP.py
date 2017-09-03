#coding=utf-8
"""
author:luchi
date:2017/9/2
"""
import numpy as np
from GridWorld import GridWorldEnv


class DP(object):


    """
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3
    """
    def policy_evaluation(self,policy,env,discount_factor=1.0, theta=0.00001,max_iter_num=1000):

        delta =0.0
        V = np.zeros(env.nS)
        iter_count=0
        while True:
            new_V = np.zeros(env.nS)
            for s in range(env.nS):
                V_x = 0.0
                for idx,prob in enumerate(policy[s]):
                    for trans_prob,next_state,reward,isFinite in env.P[s][idx]:
                        V_x += prob*trans_prob*(reward+discount_factor*V[next_state])

                new_V[s]=V_x
                delta = max(delta, np.abs(V_x - V[s]))
            iter_count+=1
            if delta<theta or iter_count>max_iter_num:
                break
            else:
                V = new_V
        return np.array(V)

    def select_biggest(self,arr):
        max_num = max(arr)
        max_arg=[]
        for idx,num in enumerate(arr):
            if np.abs(num-max_num)<1e-10:
                max_arg.append(idx)
        return max_arg

    def policy_iteration(self,policy,env,discount_factor=1.0):

        while True:
            V = self.policy_evaluation(policy,env)
            policy_stable = True
            for s in range(env.nS):
                policy_before = self.select_biggest(policy[s])
                action_values = np.zeros(env.nA)
                for a in range(env.nA):
                    for trans_prob,next_state,reward,isFinite in env.P[s][a]:
                        action_values[a]+=trans_prob*(reward+discount_factor*V[next_state])
                policy_cur = self.select_biggest(action_values)

                if not policy_before==policy_cur:
                    policy_stable=False
                new_policy = np.zeros(env.nA)
                for p in policy_cur:
                    new_policy[p]=1.0/len(policy_cur)
                policy[s]=new_policy
            if policy_stable:
                return policy,V
    def value_iteration(self,env,theta=0.0001,discount_factor=1.0):

        V = np.zeros(env.nS)
        policy=np.zeros([env.nS,env.nA])
        while True:
            delta=0
            update_V=np.zeros(env.nS)
            for s in range(env.nS):
                V_x = np.zeros(env.nA)
                for a in range(env.nA):
                    for trans_prob,next_state,reward,isFinite in env.P[s][a]:
                        V_x[a]+=trans_prob*(reward+discount_factor*V[next_state])
                policy_cur = self.select_biggest(V_x)
                new_policy = np.zeros(env.nA)
                for p in policy_cur:
                    new_policy[p]=1.0/len(policy_cur)
                policy[s]=new_policy
                bestAction = np.max(V_x)
                update_V[s]=bestAction
                delta = max(delta, np.abs(bestAction - V[s]))

            V = update_V
            if delta<theta:
                break
        return policy,V






def test_dp():
    dp = DP()
    env = GridWorldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    # V = dp.policy_evaluation(random_policy,env)
    # policy,V = dp.policy_iteration(random_policy,env)
    policy,V = dp.value_iteration(env)
    print policy
    print V.reshape([4,4])

if __name__=='__main__':
    test_dp()







