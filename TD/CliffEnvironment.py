#coding=utf



class CliffEnvironment(object):

    def __init__(self):
        self.width=12
        self.height=4
        self.move=[[0,1],[0,-1],[-1,0],[1,0]] #up,down,left,right
        self.nA=4
        self._reset()



    def _reset(self):
        self.x=0
        self.y=0
        self.end_x=11
        self.end_y=0
        self.done=False



    def observation(self):
        return tuple((self.x,self.y)),self.done
    def clip(self,x,y):
        x = max(x,0)
        x = min(x,self.width-1)
        y = max(y,0)
        y = min(y,self.height-1)
        return x,y
    def _step(self,action):
        self.done = False
        self.x+=self.move[action][0]
        self.y+=self.move[action][1]
        self.x,self.y=self.clip(self.x,self.y)

        if self.x>=1 and self.x<=10 and self.y==0:
            reward=-100
            self._reset()
        elif self.x==self.width-1 and self.y==0:
            reward=0
            self.is_destination=True
            self.done=True
        else:
            reward=-1
        return tuple((self.x,self.y)),reward,self.done







