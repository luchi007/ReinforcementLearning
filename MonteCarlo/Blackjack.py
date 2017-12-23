#coding=utf-8
import numpy as np


deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

np.random.seed(3)

#random choice a card from deck
def draw_card():
    choice = np.random.randint(0,len(deck))
    return deck[choice]

#choose two card at the begining
def draw_hand():
    return [draw_card(),draw_card()]

#define whether player or dealer has a usable ace
def usable_ace(hand):
    return (1 in hand) and (sum(hand)+10<=21)

#define the total sum of hand
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand)+10
    else:
        return sum(hand)
#whether bust
def is_bust(hand):
    return sum_hand(hand)>21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

#whether natural
def is_natural(hand):
    return hand==[1,10]



def cmp(a,b):
    return int(a>b)-int(a<b)



class Blackjack(object):
    def __init__(self,natural=False):
        self.natural = natural
        self.nA=2
        self._reset()



    def observation(self):
        return (sum_hand(self.player),self.dealer[0],usable_ace(self.player))

    def _reset(self):

        self.dealer = draw_hand()
        self.player = draw_hand()

        while sum_hand(self.player)<12:
            self.player.append(draw_card())

        return self.observation()

    def _step(self,action):
        """
        :param action: 1 means hit, 0 means stick
        :return:
        """
        done=False
        if action:
            self.player.append(draw_card())
            if is_bust(self.player):
                done=True
                reward=-1
            else:
                done = False
                reward = 0
        else:
            done=True
            while(sum_hand(self.dealer)<17):
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self.observation(),reward,done



