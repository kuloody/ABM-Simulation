import random
#import jenkspy
import pymc3 as pm
from  random import shuffle
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import pandas as pd
import scipy.stats as stats
import numpy as np
from statistics import stdev
import statistics
import math
from random import random
import pickle
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from DataBreak import *
desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',20)
from sklearn.model_selection import train_test_split

# defining the sigmoid function
import numpy as np
#Define Sigmoid function
def sig(x,a):
 a=(a/2)
 return 1/(1 + np.exp(-x))-a
 
#For Batchrun Group Classes
def data_input(data, key):
    print('Here is the data type',type(data))
    if type(data) is dict:
        ExtractedData = data[key]
        print('Here is the Exctracted dict item', ExtractedData)
 #       for item in data.values():
 #           ExtractedData = (item[:])
    else:
       ExtractedData = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS-SAMPLE.csv')

    return ExtractedData


#Define seating arrangment function
def coordenates(x,y):
    x, y = x,y
    if x % 4 == 0:
        #print('this is the coordenates function x, y is:',x,y)
        #coordenates(model)
        x+=1
        if x == 10:
            y+=1
            x= 0
        #coordenates(model)
    return x, y

def spaceCoordenates(height, width):
    gridSize = height * width
    counter = 0
    spaceSeats = []

    for x in range(width):
          #print('x is ', x)
          if x % 4 == 0:
            for y in range(height):
             if y % 2 == 0:
              spaceSeats.append((x,y))

          if counter == gridSize:
                break

    return spaceSeats
def seating(model):

    spaceSeats = spaceCoordenates(model.height, model.width)
    x, y = model.grid.find_empty()
    coordenates = x,y
    for items in spaceSeats:
        if coordenates == items:
            seating(model)
        else:
            return coordenates


#Defining shuffle function for agent position change
def shuffle_pos(list):
    shuffled_list = list.copy()
    shuffle(shuffled_list)
    return shuffled_list
def updateSeating(model):
    coords = []
    for agent in model.schedule.agents:
     coords.append(agent.pos)
    shuffled_coords=shuffle_pos(coords)

    for agent, i in zip (model.schedule.agents, shuffled_coords):
         model.grid.remove_agent(agent)
         agent.pos = i


    for agent in model.schedule.agents:
        model.grid.place_agent(agent, agent.pos)
  #  for agent in model.schedule.agents:
 #       x, y = seating(model)
#        model.grid.place_agent(agent, (x, y))


def compute_ave(model):
    agent_maths = [agent.End_maths for agent in model.schedule.agents]
    x = sum(agent_maths)
    N = len(agent_maths)
    B = x / N
    #print('the AVARAGE of end math', B, agent_maths)
    return B


def compute_ave_disruptive(model):
    agent_disruptiveTend = [agent.disruptiveTend for agent in model.schedule.agents]
    #print('Calculate disrubtive tend original', agent_disruptiveTend)
    # B = statistics.mean(agent_disruptiveTend)
    B = np.mean(agent_disruptiveTend)
    #print('Calculate disrubtive tend after mean', agent_disruptiveTend)
    #print('the AVARAGE of disruptive', B, agent_disruptiveTend)
    return B


def compute_zscore(model, x):
    agent_Inattentiveness = [agent.Inattentiveness for agent in model.schedule.agents]
    #print('Calculate variance', agent_Inattentiveness)
    SD = stdev(agent_Inattentiveness)
    mean = statistics.mean(agent_Inattentiveness)
    zscore = (x - mean) / SD
    return zscore


def compute_SD(model, x):
    agent_disruptiveTend = [agent.disruptiveTend for agent in model.schedule.agents]
    #print('Calculate variance', agent_disruptiveTend)
    b = [float(s) for s in agent_disruptiveTend]
    SD = stdev(b)
    mean = statistics.mean(b)
    zscore = (x - mean) / SD
    if zscore > 1:
        return 1
    else:
        return 0

#normalize agent ability variable
def normal(agent_ability, x):

    minValue = min(agent_ability)
    maxValue = max(agent_ability)
    rescale = (x - minValue) / maxValue - minValue
    # We want to test rescaling into a different range [1,20]
    a = 1
    b = 2
    rescale = ((b - a) * (x - minValue) / (maxValue - minValue)) + a
    return rescale


def gen_random():
    arr1 = np.random.randint(0, 21, 14)
    arr2 = np.random.randint(21, 69, 14)
    mid = [20, 21]
    i = ((np.sum(arr1 + arr2) + 41) - (20 * 30)) / 69
    decm, intg = math.modf(i)
    args = np.argsort(arr2)
    arr2[args[-70:-1]] -= int(intg)
    arr2[args[-1]] -= int(np.round(decm * 69))
    return np.concatenate((arr1, mid, arr2))
 
#Prediction using LR pre-trained model
def predictioin(features):

    data = pd.read_csv('/home/zsrj52/Downloads/SimClass/SimClassDataClassificationGrowthRateType3.csv')
    dataConvert = data.to_numpy()
    X = dataConvert[:,[2,4,5]]
    y = dataConvert[:,20]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(features)
    return y_pred

class SimClassAgent(Agent):
    # 1 Initialization
    def __init__(self, pos, model, agent_type, id, Inattentiveness, Hyperactivity, Impulsiveness, math,Start_Reading,End_Reading,Start_Vocabulary,age,fsm, IDACI,  ability,Classroom_ID):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.id = id
        self.Inattentiveness = Inattentiveness
        self.Hyperactivity = Hyperactivity
        self.Impulsiveness = Impulsiveness
        self.Start_maths = math
        self.End_maths = math
        self.Start_Reading=Start_Reading
        self.End_Reading = End_Reading
        self.Start_Vocabulary = Start_Vocabulary
        self.age = age
        self.fsm = fsm
        self.IDACI = IDACI

        self.ability = ability
        self.Classroom_ID = Classroom_ID

        self.agent_state = self.random.randint(2, 8)
        #self.agent_random = random.random()
        self.greenState = 0
        self.redState = 0
        self.yellowState = 0
        self.disrubted = 0
        self.countLearning = 0
        self.disruptiveTend = Inattentiveness
        #Adding neighbours and Sigmoid
        self.neighbours = 0
        self.Sigmoid = 0
        self.minute_counter = 0

        #Create Array of agent features
        self.agentAttr = np.array([self.Start_maths, self.Inattentiveness, self.Hyperactivity])
        self.agentAttr = self.agentAttr.reshape((1, -1))
        self.growthRate = predictioin(self.agentAttr)
        self.singleValueGrowth = float(self.growthRate[0])

        #load the trained model for prediction
        #loaded_model = pickle.load(open('/home/zsrj52/PycharmProjects/SimClass/prediction_model.sav', 'rb'))
        #self.growth_pred = loaded_model.predict(self.agentAttr.tolist()).tolist()




    def neighbour(self):
        neighbourCount = 0
        red = 0
        green = 0
        yellow = 0
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            neighbourCount += 1
            if neighbor.type == 3:
                red += 1
            elif neighbor.type == 2:
                yellow += 1
            else:
                green += 1

        return neighbourCount, red, yellow, green

    def step(self):

        print('Step number:',self.model.schedule.steps)
        #print('Count learning value:', self.countLearning)
        #print('Agent position', self.pos)
        #print('check agent vars',self.model.datacollector.get_agent_vars_dataframe())
        self.stateCurrent()
        '''''
        #Change agent state per defined minutes
        if self.minute_counter ==self.model.State_Minutes:
            self.stateCurrent()
            print('#slider',self.model.State_Minutes)
            print('self.model.minute_counter',self.minute_counter)
            self.minute_counter = 0
        '''''

        #print('self.model.minute_counter',self.minute_counter)
        self.minute_counter +=1
        if self.type==1:
           self.Green_State()
        if self.type == 2:
            self.Yellow_State()
        if self.type == 3:
            self.Red_State()

        return

# defineing a unified state function
    def stateCurrent(self):
        count, red, yellow, green = self.neighbour()
        randomVariable = self.random.uniform(-1, 1)

        y = (red+yellow-green+self.type+self.Inattentiveness+self.Hyperactivity+randomVariable)
        self.Sigmoid = sig(y,self.model.Nthreshold)
        self.neighbours = y
        #print('student Inattentiveness & Hyperactivity:',self.Inattentiveness,self.Hyperactivity)
        #print('the total of student parameters: ',y)
        #print('the value of random variable',randomVariable)
        #print('the result of sigmoid:',self.Sigmoid)

        propability = self.random.uniform(0.98, 1)

        #if x < self.model.Nthreshold:
        if y <= self.model.Nthreshold:

            #if self.Sigmoid <= 0.6:
                #print( 'the value of SIgmoid is $',self.Sigmoid,y )
            self.type = 1
            self.model.learning += 1
            self.countLearning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return 1
        else :
            if self.Impulsiveness == 0:
                self.type = 2
                self.disrubted += 0.5
                self.countLearning += 0.3
                self.redState = 0
                self.yellowState += 1
                self.greenState = 0
                self.set_start_math()
                Intercept = 1
                #self.End_maths = (3.33 * Intercept + 0.43 * self.Start_maths + 0.10 * self.Start_Reading + 0.45 * self.Start_Vocabulary + -0.01 * self.Inattentiveness + 0.81 * self.age + self.random.randint(-1, 0)+self.ability)/50

                return 1
            else :
                self.type = 3
                self.model.distruptive += 1
                self.countLearning += 0.1
                self.disrubted += 1
                self.redState += 1
                self.yellowState = 0
                self.greenState = 0
                self.set_start_math()
                Intercept = 1
                #self.End_maths = (3.33 * Intercept + 0.43 * self.Start_maths + 0.10 * self.Start_Reading + 0.45 * self.Start_Vocabulary + -0.01 * self.Inattentiveness + 0.81 * self.age + self.random.randint(-1, 0)+self.ability)/80

            return 1
    def Green_State(self):
                self.type = 1
                self.model.learning += 1
                self.countLearning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1

    def Yellow_State(self):
                self.type = 2
                self.disrubted += 0.5
                self.countLearning += 0.3
                self.redState = 0
                self.yellowState += 1
                self.greenState = 0
                self.set_start_math()

    def Red_State(self):
            self.type = 3
            self.model.distruptive += 1
            self.countLearning += 0.1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            self.set_start_math()

    def redStateCange(self):
        count, red, yellow, green = self.neighbour()

        if red > 2 and self.type == 2:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

        if red > self.agent_state  and self.disruptiveTend > compute_ave_disruptive(self.model):
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        # if Inattentiveness is on and quality is low
        if self.model.Inattentiveness == 1 and self.model.quality <= self.agent_state + 1 and self.Inattentiveness > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1
        # If both is high and student is disruptive
        if self.model.Inattentiveness == 1 and self.model.control <= self.agent_state + 1 and self.Inattentiveness > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

        if self.model.hyper_Impulsive == 1 and self.model.control <= self.agent_state and self.Hyperactivity > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1
        if (self.model.control or self.model.quality) < self.agent_state and self.Hyperactivity > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1



    def yellowStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.model.Inattentiveness == 1 and self.model.quality >= self.agent_state and self.Inattentiveness <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        #student is disruptive but inattintiveness is off
        if self.model.Inattentiveness == 0 and self.Inattentiveness > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if compute_SD(self.model,
                      self.disruptiveTend) and self.model.control >= self.agent_state and self.Hyperactivity <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.model.quality > self.agent_state and self.Inattentiveness > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        # if control is less than student state
        if self.model.control <= self.agent_state and self.type == 1:
            self.type = 2
            if self.model.learning > 0:
                self.model.learning -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        # At general if control is high turn into passive
        if self.model.control > self.agent_state and self.Hyperactivity > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1



        # Change state based on majority of neighbours' color and agent's current color state

    @property
    def greenStateCange(self):

        count, red, yellow, green = self.neighbour()

        if green > 5 and self.type == 2:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return 1


        #if self.model.Inattentiveness != 0 or self.model.quality <= self.agent_state or self.Inattentiveness >= self.agent_state:
        if self.model.Inattentiveness == 0 and self.model.quality > self.agent_state:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1

        if self.model.hyper_Impulsive == 0 and self.model.control > self.agent_state >= self.Hyperactivity:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1

        if self.model.hyper_Impulsive == 0 and self.agent_state >= self.Hyperactivity and self.model.quality > self.agent_state:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1
        if self.type == 2 and self.Inattentiveness <= self.agent_state:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1
        #return
      #  self.type = 1
      #  self.model.learning += 1
      #  self.set_start_math()
     #   self.redState = 0
      #  self.yellowState = 0
     #   self.greenState += 1
      #  return 1

    def neighbourState(self):
        count, red, yellow, green = self.neighbour()
        # calculate the probability of each colour
        Pturn_red = red / count
        Pturn_green = green / count
        Pturn_yellow = yellow / count

        if self.type == 3:
            Pturn_red += 0.2
        elif self.type == 2:
            Pturn_yellow += 0.2
        else:
            Pturn_green += 0.2
        colour = max(Pturn_red, Pturn_green, Pturn_yellow)
        if Pturn_red == colour:
            self.type = 3
            #self.model.distruptive += 1
            return
        if Pturn_yellow == colour:
            self.type = 2
            return
        if Pturn_green == colour:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            return

    def changeState(self):

        # Change to attentive (green) teaching quality or control is high and state is passive for long
        if (self.model.quality or self.model.control) > self.agent_state and self.yellowState >= self.agent_state:
            self.type = 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1

        if self.Hyperactivity < self.agent_state and self.model.control >= self.agent_state and self.yellowState > self.agent_state:
            self.type = 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()

        if self.model.control < self.agent_state and self.yellowState > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

            return 1
        # Similar to above but red for long
        if self.Hyperactivity > self.agent_state < self.redState and self.model.control >= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.Inattentiveness < self.agent_state < self.redState and self.model.quality <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.Inattentiveness < self.agent_state and self.model.quality <= self.agent_state and self.yellowState > self.agent_state:
            self.type = 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()

            return 1

        if self.Inattentiveness > self.agent_state and self.model.quality > self.agent_state and self.redState > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.Hyperactivity > self.agent_state - 1 and self.model.control > self.agent_state - 1 and self.redState > self.agent_state-1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if (self.model.control and self.model.quality) > self.agent_state and self.redState > self.agent_state:
            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1

        if self.Hyperactivity <= self.agent_state and self.model.quality <= self.agent_state and self.redState > self.agent_state -1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.Hyperactivity <= self.agent_state - 1 and self.redState > self.agent_state -1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.Inattentiveness < self.agent_state and self.redState > self.agent_state -1:
            self.type = 2
            self.disrubted += 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.yellowState > 2:
            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1




    def set_start_math(self):
        # Increment the learning counter
        #self.countLearning += 1
        #Transform the disrupted counter
        Disrupted= self.disrubted * self.random.uniform(0.00,0.002)
        Intercept = 1
        MaxScore = 69
        # Evalue = MaxScore /ln 8550
        # Evalue = 7.621204857
        Evalue = 6.074873437
        Scaled_Smath = (2.718281828 ** self.Start_maths) ** (1 / Evalue)
        total_learn = self.countLearning + Scaled_Smath
        end_maths = (Evalue * math.log(total_learn))
        #self.End_maths = 3.33 * Intercept + 0.43 * self.Start_maths + 0.10 * self.Start_Reading + 0.45 * self.Start_Vocabulary + (-0.01 * self.Inattentiveness) + 0.81 * self.age #+ self.random.randint(-1, 0)#+self.ability
        #self.End_maths = (7.621204857 * math.log(end_maths))
        #self.End_maths = self.BayModel()
        self.End_maths = self.LinearRegressionModel() - Disrupted + self.random.randint(-2, 2)


        # Scale Smath before using it to calculate end math score
        # 8550t = 7.621204857 for the scale of 69
        # 8550t =
        #random () generates a float number between 1 nd 0 of a normal distribution

        #Old scale math:
        Scaled_Smath = (2.718281828 ** self.Start_maths) ** (1 / 7.621204857)
        total_learn = self.countLearning + Scaled_Smath
        #self.End_maths = (7.621204857 * math.log(total_learn)) + self.ability #learn minutes transfer formula
        #self.End_maths = self.Start_maths * (1.001275 ** self.countLearning) #old growthrate
        MaxGainedScore = 69 - self.Start_maths
        n= math.log(MaxGainedScore)/math.log(8550)
        #Scaled_Smath = (2.718281828 ** self.Start_maths) ** (1 / n)
        #self.End_maths = (self.countLearning ** n) + self.Start_maths
        if self.fsm == 1:
            fsmVariable = self.random.randint(-1, 0)
        else:
            fsmVariable = self.random.randint(0, 1)

        #Formula from Baysian regrission
    def BayModel (self):
        with open('/home/zsrj52/Downloads/SimClass/LinearModel.sav', 'rb') as buff:
                data = pickle.load(buff)
        model = data['model']
        trace = data['trace']
        X_shared = data['X_train']

        with model:
            post_pred = pm.sample_posterior_predictive(trace)

            var_dict = {}
        for variable in trace.varnames:
            var_dict[variable] = trace[variable]

        # Results into a dataframe
        var_weights = pd.DataFrame(var_dict)

        # Standard deviation of the likelihood
        sd_value = var_weights['sd'].mean()

        # Actual Value
        data = [self.Start_maths,self.Start_Reading,self.Start_Vocabulary,self.Inattentiveness,self.age]
        test_observation =pd.DataFrame(data)

        print('test observation',test_observation)

        # Align weights and test observation
        var_weights = var_weights[test_observation.index]

        # Means for all the weights
        var_means = var_weights.mean(axis=0)

        # Location of mean for observation
        mean_loc = np.dot(var_means, test_observation)
            # Location of mean for observation
        mean_loc = np.dot(var_means, test_observation)

        # Estimates of grade
        estimates = np.random.normal(loc = mean_loc, scale = sd_value,
                                 size = 1000)
        return estimates


        #self.End_maths = (self.s_math * (1.00008851251853 ** self.countLearning)) * self.ability #- compute_zscore(self.model,self.Inattentiveness)
        #self.End_maths = (self.s_math * (self.singleValueGrowth ** self.countLearning)) * self.ability - compute_zscore(self.model,self.Inattentiveness) + self.random.randint(0, 1) + fsmVariable

    def LinearRegressionModel(self):
        # load the model from disk
        filename = '/home/zsrj52/Downloads/SimClass/LinearModel.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        #9 Features
        #Pdata = [self.Start_maths,self.Start_Reading,self.Start_Vocabulary,self.Inattentiveness,self.age,self.Start_Vocabulary,self.fsm,self.IDACI,self.Impulsiveness]
        Pdata = [self.Start_maths,self.Start_Reading,self.Start_Vocabulary,self.Inattentiveness,self.age]
        Pdata = np.array(Pdata)
        pred = loaded_model.predict(Pdata.reshape(1, -1))
        #pred = pred.astype(int)
        print('Linear Regrission Prediction', pred[0],type(pred))
        return pred[0]
    def get_type(self):
        return self.type

    def set_disruptive_tend(self):

        self.initialDisrubtiveTend = compute_zscore(self.model, self.Inattentiveness)

        print("HERE AFTER Z SCORE", self.initialDisrubtiveTend)
        if self.model.schedule.steps == 0:
            self.model.schedule.steps = 1

        self.disruptiveTend = (((self.disrubted / self.model.schedule.steps) - (
                self.countLearning / self.model.schedule.steps)) + self.initialDisrubtiveTend)



class SimClass(Model):


    def __init__(self,data, key=1, height=11, width=11, quality=1, Inattentiveness=0, control=3,Seating=1,State_Minutes=1, hyper_Impulsive=0, AttentionSpan=0, Nthreshold = 3, NumberofGroups=2):

        self.height = height
        self.width = width
        self.quality = quality
        self.Inattentiveness = Inattentiveness
        self.control = control
        self.hyper_Impulsive = hyper_Impulsive
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.AttentionSpan = AttentionSpan
        self.Nthreshold = Nthreshold
        self.NumberofGroups = NumberofGroups
        self.Seating = Seating
        self.State_Minutes = State_Minutes

        self.learning = 0
        self.distruptive = 0
        self.schoolDay = 0
        self.daycounter = 0

        self.redState = 0
        self.yellowState = 0
        self.greenState = 0
        self.coords=[]


        #Load data

        #data = pd.read_csv('/home/zsrj52/Downloads/SimClass/DataSampleNochange.csv')

        #data = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS-SAMPLE.csv')
        #data = data
        data = data_input(data,key)

        # Take agent variables from the data

        maths = data['Start_maths'].to_numpy()
        ability_zscore = stats.zscore(maths)
        Inattentiveness = data['Inattentiveness'].to_numpy()
        Hyperactivity = data['Hyperactivity'].to_numpy()
        Impulsiveness = data['Impulsiveness'].to_numpy()
        Start_Reading = data['Start_Reading'].to_numpy()
        End_Reading = data['End_Reading'].to_numpy()
        age = data['End_age'].to_numpy()
        Start_Vocabulary = data['Start_Vocabulary'].to_numpy()
        fsm = data['FSM'].to_numpy()
        id = data['ID'].to_numpy()
        IDACI = data['IDACI'].to_numpy()
        Classroom_ID = data['Classroom_ID'].to_numpy()
        coords=[]

        # Set up agents

        counter = 0
 #       for cell in self.grid.coord_iter():
        for x in range(width):
          #print('x is ', x)
          #print('size is', data.shape[0])
          if x % 4 == 0:
            continue
          for y in range(height):
            if y % 2 == 0:
             continue

            if counter == data.shape[0]:
                break
            #x,y =coordenates(width,height)

            #x = cell[1]
            #y = cell[2]

            #Place agents in different postions each run
 #           x, y = self.grid.find_empty()


            # Initial State for all student is random
            agent_type = self.random.randint(1, 3)
#            ability = ability_zscore[counter]
            ability = normal(ability_zscore, ability_zscore[counter])

            # create agents form data
            agent = SimClassAgent((x, y), self, agent_type, id[counter], Inattentiveness[counter], Hyperactivity[counter],Impulsiveness[counter],
                                  maths[counter],Start_Reading[counter],End_Reading[counter],Start_Vocabulary[counter],age[counter],fsm[counter],IDACI[counter], ability, Classroom_ID)
            # Place Agents on grid
            #x, y = self.grid.find_empty()
            self.grid.place_agent(agent, (x, y))
            coords.append([x,y])
            print('agent pos:', x, y)
            self.schedule.add(agent)
            counter += 1


        # Collecting data while running the model
        self.datacollector = DataCollector(
            model_reporters={"Distruptive Students": "distruptive",
                             "Learning Students": "learning",
                             "School Day": "schoolDay",
                             "Average End Math": compute_ave,
                             "disruptiveTend": compute_ave_disruptive
                             },
            # Model-level count of learning agent
            agent_reporters={"x": lambda a: a.pos[0], "y": lambda a: a.pos[1], "id":"id",
                             "Hyperactivity": "Hyperactivity", "Start_maths": "Start_maths", "Start_Reading": "Start_Reading",
                             "End_maths": "End_maths", "End_Reading": "End_Reading","Start_Vocabulary":"Start_Vocabulary","Inattentiveness": "Inattentiveness","End_age":"age", "ability": "ability",
                             "LearningTime": "countLearning", "disruptiveTend": "disruptiveTend", "Sigmoid":"Sigmoid", "neighbours":"neighbours","disrubted":"disrubted","Classroom_ID":"Classroom_ID"})

        self.running = True

    def step(self):


        self.learning = 0  # Reset counter of learing and disruptive agents
        self.distruptive = 0
        Daycounter = self.schedule.time
        self.daycounter+=1


        if self.daycounter == 45:
            self.schoolDay +=1
            self.daycounter = 0
            if self.Seating == 1 :
             updateSeating(self)



        self.datacollector.collect(self)
        self.schedule.step()
        Daycounter =+1

        # collect data
        self.datacollector.collect(self)

        if self.schedule.steps == 8550 or self.running == False:
            self.running = False
            dataAgent = self.datacollector.get_agent_vars_dataframe()
            dataAgent.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-120/all-linearRegrission-9FEATURE.csv')
