#coding=utf-8

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import plotting

env = gym.envs.make("MountainCar-v0")
#采样数据
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

#使用RBF核函数进行特征转换
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))

class Estimator(object):
    def __init__(self):
        self.models=[]
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.feature_state(env.reset())],[0])
            self.models.append(model)

    def predict(self,s,a=None):
        s=self.feature_state(s)
        if a:
            return self.models[a].predict([s])[0]
        else:
            return [self.models[m].predict([s])[0] for m in range(env.action_space.n)]

    def update(self,s,a,target):
        s=self.feature_state(s)
        self.models[a].partial_fit([s],[target])

    def feature_state(self,s):

        return featurizer.transform(scaler.transform([s]))[0]

def make_epsilon_greedy_policy(estimator,nA,epsilon):

    def epsilon_greedy_policy(observation):

        best_action = np.argmax(estimator.predict(observation))
        A =np.ones(nA,dtype=np.float32)*epsilon/nA
        A[best_action] += 1-epsilon
        return A

    return epsilon_greedy_policy


def Q_learning_with_value_approximation(env,estimator,epoch_num
                                        ,discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(epoch_num),
    #     episode_rewards=np.zeros(epoch_num))
    for i_epoch_num in range(epoch_num):

        policy = make_epsilon_greedy_policy\
            (estimator,env.action_space.n,epsilon*epsilon_decay**i_epoch_num)
        state = env.reset()

        for it in itertools.count():

            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state,reward,done,_=env.step(action)
            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            estimator.update(state, action, td_target)

            # stats.episode_rewards[i_epoch_num] += reward
            # stats.episode_lengths[i_epoch_num] = it
            print("\rStep {} @ Episode {}/{}".format(it, i_epoch_num + 1, epoch_num))

            if done:
                print it
                break
            state = next_state


estimator=Estimator()
Q_learning_with_value_approximation(env, estimator, 100, epsilon=0.0)
plotting.plot_cost_to_go_mountain_car(env, estimator)










