class State:

    def __init__(self):
        self.name = 'Abstract microstate'

    def __str__(self):
        return self.name

class RandomWalk1D(State):

    def __init__(self):
        self.name = 'Random Walk State'
        self.occupied = false

    def linkPrior(self, pre):
        self.predecessor = pre

    def linkFollow(self, suc):
        self.successor = suc

    def setRates(self, forward, backward):
        self.f_rate = forward
        self.b_rate = backward
        self.rate_param = self.getRateParameter()

    def getRateParameter(self):
        param = 0
        if self.f_rate != None:
            param += self.f_rate
        if self.b_rate != None:
            param += self.b_rate
        return param

    def jump(self):
        if self.absorbing:
            return self
        if self.f_rate == None:
            return self.predecessor
        elif self.b_rate == None:
            return self.successor
        else:
            fraction = self.f_rate/self.rate_param
            if(random.random() >= fraction):
                return self.predecessor
            else:
                return self.successor


