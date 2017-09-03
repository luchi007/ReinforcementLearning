#coding=utf-8
import numpy as np
"""
author:luchi
date:2017/9/2
description:gridworld environment
"""

class GridWorldEnv(object):


    def __init__(self,shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.nS = np.prod(shape)
        self.nA = 4 # four directions
        MAX_X = shape[0]
        MAX_Y = shape[1]

        P={}
        grid = np.arange(self.nS).reshape(shape)

        it = np.nditer(grid,flags=['multi_index'])
        #move directions
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        while not it.finished:

            state = it.iterindex
            x,y = it.multi_index
            # print state
            # print(x,y)
            P[state] = {a:[] for a in range(self.nA)}
            is_terminal_state = lambda state: state==0 or state==self.nS-1
            reward = lambda state : 0.0 if is_terminal_state(state) else -1.0
            if is_terminal_state(state):
                P[state][UP] = [(1.0,state,reward(state),True)]
                P[state][RIGHT] = [(1.0,state,reward(state),True)]
                P[state][DOWN] = [(1.0,state,reward(state),True)]
                P[state][LEFT] = [(1.0,state,reward(state),True)]
            else:
                Up_Grid = state if x==0 else state-MAX_Y
                Right_Grid = state if y==MAX_Y-1 else state+1
                Down_Grid = state if x==MAX_X-1 else state+MAX_Y
                Left_Grid = state if y==0 else state-1

                P[state][UP] = [(1.0,Up_Grid,reward(Up_Grid),is_terminal_state(Up_Grid))]
                P[state][RIGHT] = [(1.0,Right_Grid,reward(Right_Grid),is_terminal_state(Right_Grid))]
                P[state][DOWN] = [(1.0,Down_Grid,reward(Down_Grid),is_terminal_state(Down_Grid))]
                P[state][LEFT] = [(1.0,Left_Grid,reward(Left_Grid),is_terminal_state(Left_Grid))]
            # print P[state]
            it.iternext()
            self.P = P

grid = GridWorldEnv()