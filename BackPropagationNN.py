# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string
from numba import jit, cuda 
import time
from timeit import default_timer as timer 

random.seed(0)
myDataError = []

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    e = math.tanh(x)
    return e

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NeuralNetwork:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
            #print("Return: " + str(self.ao[:]))
            self.update(p[0])
            return self.ao[:]

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=50000, N=0.11, M=0.13):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N/(1+(i*(1.0/5))), M) #/(1+(i*(1.0/3)))
                
            if i % 200 == 0:
                print(str(i) + ": " + 'error %-.5f' % error)
            myDataError.append(error)
                
    def SaveWeights(self):
        next_line="\n"
        file = open("Weights_wi.txt", "w")
        for i in range(self.ni):
            for j in range(self.nh):
                file.writelines(str(self.wi[i][j]))
                file.writelines(next_line)
        file.close()
        
        file = open("Weights_wo.txt", "w")
        for w in range(self.nh):
            for k in range(self.no):
                file.writelines(str(self.wo[w][k]))
                file.writelines(next_line)
        file.close()
        
    def SaveError(self):
        next_line="\n"
        file = open("errorNN.txt", "w")
        for i in range(50000):
            file.writelines(str(myDataError[i]))
            file.writelines(next_line)
        file.close()
        
    def ReadWeights(self):
        index = 0
        file = open("Weights_wi.txt", "r")
        my_list = file.readlines()
        file.close()
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = float(my_list[index])
                index = index + 1

        index = 0
        file = open("Weights_wo.txt", "r")
        my_list = file.readlines()
        file.close()
        for w in range(self.nh):
            for k in range(self.no):
                self.wo[w][k] = float(my_list[index])
                index = index + 1
                
    def ReadDataToTeach(self):
        index = 0
        file = open("MyDataToTeaching.txt", "r")
        my_list = file.readlines()
        file.close()
        data_to_teach = []
        print(len(my_list))
        for a in range(int(len(my_list)/22)):
            read_input_param = []
            read_output_param = []
            for param_input in range(2*a+20*a, 2*a+20*a+20, 1):
                read_input_param.append(float(my_list[param_input]))
            for param_output in range(2*a+20*a+20, 2*a+20*a+22, 1):
                read_output_param.append(float(my_list[param_output]))
            data_to_teach.append([read_input_param] + [read_output_param])
        
        return data_to_teach
                
def demo():
    # Teach network 
    start = timer() 
    pat = [
        [[0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.007063786044343989, 0.0017534669925282527, 0.634, 0.211, 0.740, 0.992, -0.100, 0.508, 0.834, 0.083, 0.7, 0.712, -0.095, 0.6]]
        ]
    
    # create a network with two input, two hidden, and one output nodes
    n = NeuralNetwork(20, 14, 2)
    data_to_teach = n.ReadDataToTeach()
    print("Data to teach: " + str(data_to_teach))
    # train it with some patterns
    n.train(data_to_teach)
    # test it
    #n.ReadWeights()
    n.test(pat)
    print("time:", timer()-start)
    n.SaveWeights()
    n.SaveError()

if __name__ == '__main__':
    demo()
