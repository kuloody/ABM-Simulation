import random
import jenkspy
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
def sig(x):
 return 1/(1 + np.exp(-x))

def compute_ave(model):
    agent_maths = [agent.e_math for agent in model.schedule.agents]
    x = sum(agent_maths)
    N = len(agent_maths)
    B = x / N
    print('the AVARAGE of end math', B, agent_maths)
    return B


def compute_ave_disruptive(model):
    agent_disruptiveTend = [agent.disruptiveTend for agent in model.schedule.agents]
    print('Calculate disrubtive tend original', agent_disruptiveTend)
    # B = statistics.mean(agent_disruptiveTend)
    B = np.mean(agent_disruptiveTend)
    print('Calculate disrubtive tend after mean', agent_disruptiveTend)
    print('the AVARAGE of disruptive', B, agent_disruptiveTend)
    return B


def compute_zscore(model, x):
    agent_behave = [agent.behave for agent in model.schedule.agents]
    print('Calculate variance', agent_behave)
    SD = stdev(agent_behave)
    mean = statistics.mean(agent_behave)
    zscore = (x - mean) / SD
    return zscore


def compute_SD(model, x):
    agent_disruptiveTend = [agent.disruptiveTend for agent in model.schedule.agents]
    print('Calculate variance', agent_disruptiveTend)
    b = [float(s) for s in agent_disruptiveTend]
    SD = stdev(b)
    mean = statistics.mean(b)
    zscore = (x - mean) / SD
    if zscore > 1:
        return 1
    else:
        return 0


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
    def __init__(self, pos, model, agent_type, id, behave, behave_2, math,age,fsm,  ability):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.id = id
        self.behave = behave
        self.behave_2 = behave_2
        self.s_math = math
        self.e_math = math
        self.age = age
        self.fsm = fsm

        self.ability = ability

        self.agent_state = self.random.randint(2, 8)
        #self.agent_random = random.random()
        self.greenState = 0
        self.redState = 0
        self.yellowState = 0
        self.disrubted = 0
        self.countLearning = 0
        self.disruptiveTend = behave
        #Adding neighbours and Sigmoid
        self.neighbours = 0
        self.Sigmoid = 0

        #Create Array of agent features
        self.agentAttr = np.array([self.s_math, self.behave, self.behave_2])
        self.agentAttr = self.agentAttr.reshape((1, -1))
        self.growthRate = predictioin(self.agentAttr)
        self.singleValueGrowth = float(self.growthRate[0])

        #load the trained model for prediction
        #loaded_model = pickle.load(open('/home/zsrj52/PycharmProjects/SimClass/prediction_model.sav', 'rb'))
        #self.growth_pred = loaded_model.predict(self.agentAttr.tolist()).tolist()



        #Load joblib model
        m_jlib = joblib.load('model_jlib')
        self.job_pred = m_jlib.predict(self.agentAttr.tolist()).tolist()

        print('Agent features',self.agentAttr)
        #print('Loaded model Prediction: ',self.job_pred)
        print('function model Prediction: ',self.growthRate)

    # self.greenState = 0

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
    """""
    # define the step function

    def step(self):
        #   self.disrubted += 1
        # self.changeState()
        print(self.model.schedule.steps)
        print('Agent position', self.pos)
        if self.redStateCange() == 1:
            # self.model.distruptive += 1
            self.changeState()
            #if self.type == 3:
                #self.model.distruptive += 1
            self.set_disruptive_tend()
            self.agent_state = self.random.randint(2, 6)

            return
        elif self.greenStateCange == 1:
            self.changeState()
            self.set_disruptive_tend()
            self.agent_state = self.random.randint(2, 6)

            return

        elif self.yellowStateCange() == 1:
            self.set_disruptive_tend()
            self.changeState()
            #if self.type == 3:
                #self.model.distruptive += 1
            self.agent_state = self.random.randint(2, 6)


            return

        self.agent_state = self.random.randrange(2,6)

"""
    def step(self):
        #   self.disrubted += 1
        # self.changeState()
        print('Step number:',self.model.schedule.steps)
        print('Count learning value:', self.countLearning)
        print('current emath value:', self.e_math)
        print('Agent position', self.pos)
        self.agent_state_1 = self.random.uniform(0.0,0.5)
        self.stateCurrent()
        self.changeState()
        return

# defineing a unified state function
    def stateCurrent(self):
        count, red, yellow, green = self.neighbour()
        randomVariable = self.random.uniform(-6, 6)
        y = red+yellow+green+self.type+self.behave+self.behave_2-(self.model.quality+self.model.control)-randomVariable
        self.Sigmoid = sig(y)
        self.neighbours = y

        propability = self.random.uniform(0.98, 1)

        #if x < self.model.Nthreshold:
        if self.Sigmoid < 0.95:
            #r = x / 1000
            print( 'the value of Sigmoid is $$$$$$$**$$$$$$$' ,self.Sigmoid,y,self.agent_state_1)
            if self.Sigmoid < self.agent_state_1:
                print( 'the value of SIgmoid is $$$$$$$$$$$$$$$$',self.Sigmoid,y )
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1
            else:
                self.type = 2
                self.redState = 0
                self.yellowState += 1
                self.greenState = 0
                return 1
        else:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

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
        if self.model.Inattentiveness == 1 and self.model.quality <= self.agent_state + 1 and self.behave > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1
        # If both is high and student is disruptive
        if self.model.Inattentiveness == 1 and self.model.control <= self.agent_state + 1 and self.behave > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

        if self.model.hyper_Impulsive == 1 and self.model.control <= self.agent_state and self.behave_2 > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1
        if (self.model.control or self.model.quality) < self.agent_state and self.behave_2 > self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1



    def yellowStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.model.Inattentiveness == 1 and self.model.quality >= self.agent_state and self.behave <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        #student is disruptive but inattintiveness is off
        if self.model.Inattentiveness == 0 and self.behave > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if compute_SD(self.model,
                      self.disruptiveTend) and self.model.control >= self.agent_state and self.behave_2 <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.model.quality > self.agent_state and self.behave > self.agent_state:
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
        if self.model.control > self.agent_state and self.behave_2 > self.agent_state:
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


        #if self.model.Inattentiveness != 0 or self.model.quality <= self.agent_state or self.behave >= self.agent_state:
        if self.model.Inattentiveness == 0 and self.model.quality > self.agent_state:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1

        if self.model.hyper_Impulsive == 0 and self.model.control > self.agent_state >= self.behave_2:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1

        if self.model.hyper_Impulsive == 0 and self.agent_state >= self.behave_2 and self.model.quality > self.agent_state:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1
        if self.type == 2 and self.behave <= self.agent_state:
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

        if self.behave_2 < self.agent_state and self.model.control >= self.agent_state and self.yellowState > self.agent_state:
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
        if self.behave_2 > self.agent_state < self.redState and self.model.control >= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.behave < self.agent_state < self.redState and self.model.quality <= self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.behave < self.agent_state and self.model.quality <= self.agent_state and self.yellowState > self.agent_state:
            self.type = 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()

            return 1

        if self.behave > self.agent_state and self.model.quality > self.agent_state and self.redState > self.agent_state:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if self.behave_2 > self.agent_state - 1 and self.model.control > self.agent_state - 1 and self.redState > self.agent_state-1:
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

        if self.behave_2 <= self.agent_state and self.model.quality <= self.agent_state and self.redState > self.agent_state -1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.behave_2 <= self.agent_state - 1 and self.redState > self.agent_state -1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.behave < self.agent_state and self.redState > self.agent_state -1:
            self.type = 2
            self.disrubted += 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        '''if self.redState > self.agent_state:
            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1'''

    def set_start_math(self):
        # Increment the learning counter
        self.countLearning += 1

        # Scale Smath before using it to calculate end math score
        # 8550t = 7.621204857
        # new scale of Smath
        # 8550 = 0.467666566
        #random () generates a float number between 1 nd 0 of a normal distribution
        MaxGainedScore = 69 - self.s_math
        n= math.log(MaxGainedScore)/math.log(8550)
        Scaled_Smath = (2.718281828 ** self.s_math) ** (1 / n)
        self.e_math = (self.countLearning ** n ) + self.s_math
        #Old scale math:
        #Scaled_Smath = (2.718281828 ** self.s_math) ** (1 / 7.621204857)
        total_learn = self.countLearning + Scaled_Smath
        #self.e_math = (7.621204857 * math.log(total_learn)) + self.ability #learn minutes transfer formula
        #self.e_math = self.s_math * (1.001275 ** self.countLearning) #old growthrate
        if self.fsm == 1:
            fsmVariable = self.random.randint(-1, 0)
        else:
            fsmVariable = self.random.randint(0, 1)


        #self.e_math = (self.s_math * (1.00008851251853 ** self.countLearning)) * self.ability #- compute_zscore(self.model,self.behave)
        #self.e_math = (self.s_math * (self.singleValueGrowth ** self.countLearning)) * self.ability - compute_zscore(self.model,self.behave) + self.random.randint(0, 1) + fsmVariable


    def get_type(self):
        return self.type

    def set_disruptive_tend(self):

        self.initialDisrubtiveTend = compute_zscore(self.model, self.behave)

        print("HERE AFTER Z SCORE", self.initialDisrubtiveTend)
        if self.model.schedule.steps == 0:
            self.model.schedule.steps = 1

        self.disruptiveTend = (((self.disrubted / self.model.schedule.steps) - (
                self.countLearning / self.model.schedule.steps)) + self.initialDisrubtiveTend)

    def test(self):
        print('HI I am test function ##############################')
"""
    def set_start_math(self):
        # Increment the learning counter
        self.countLearning += 1

        # Scale Smath before using it to calculate end math score
        # 8550t = 7.621204857
        #random () generates a float number between 1 nd 0 of a normal distribution

        Scaled_Smath = (2.718281828 ** self.s_math) ** (1 / 7.621204857)
        total_learn = self.countLearning + Scaled_Smath
        #self.e_math = (7.621204857 * math.log(total_learn)) + self.ability
        #self.e_math = self.s_math * (1.001275 ** self.countLearning) #old growthrate

        #self.e_math = (self.s_math * (1.00008851251853 ** self.countLearning)) * self.ability #- compute_zscore(self.model,self.behave)
        self.e_math = (self.s_math * (self.singleValueGrowth ** self.countLearning)) * self.ability - compute_zscore(self.model,self.behave) + self.random.randint(0, 1) #* self.ability #

    def set_start_math(self):
        # Increment the learning counter
        self.countLearning += 1
        #if self.model.schedule.steps == 45:
            #self.model.schoolDay +=1
        self.age = self.age + self.model.schoolDay/365
        print('type age &&&&&', self.age)

        e = 100 + 5 * np.random.randn(1)
        s = 1.0 + 0.115 * np.random.randn(1)
        S = 1.0 + 0.085 * np.random.randn(1)
        x = self.age
        n = 4.5 + 0.5 * np.random.randn(1)
        B = 100 + 5 * np.random.randn(1)
        e2 = (s + S) * B * x + ((s + S) * n * x)
        print('type e &&&&&', type(e))
        function ='e * self.age'
        function2 = '(s + S) * B * x + ((s + S) * n * x)'
        value = eval(function2) * 1.00
        print('type value &&&&&', type(value))
        self.e_math= value.tolist()
        self.e_math = self.e_math[0]
        print('type emath &&&&&', type(self.e_math))
        print(' emath &&&&&', self.e_math)

"""



class SimClass(Model):


    def __init__(self, height=6, width=6, quality=1, Inattentiveness=0, control=3, hyper_Impulsive=0, AttentionSpan=0, Nthreshold = 3, NumberofGroups=2):

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

        self.learning = 0
        self.distruptive = 0
        self.schoolDay = 0
        self.daycounter = 0
        self.redState = 0
        self.yellowState = 0
        self.greenState = 0

        #Load data

        #data = pd.read_csv('/home/zsrj52/Downloads/SimClass/DataSampleNochange.csv')
        data = pd.read_csv('/home/zsrj52/Downloads/SimClass/SampleDataFullVariables.csv')

        # Take agent variables from the data
        maths = data['s_maths'].to_numpy()
        ability_zscore = stats.zscore(maths)
        behave = data['behav1'].to_numpy()
        behav2 = data['behav2'].to_numpy()
        age = data['Age_at_start'].to_numpy()
        fsm = data['FSM'].to_numpy()
        id = data['id'].to_numpy()

        # Set up agents

        counter = 0
        for cell in self.grid.coord_iter():
            #x = cell[1]
            #y = cell[2]

            #Place agents in different postions each run
            x, y = self.grid.find_empty()

            # Initial State for all student is random
            agent_type = self.random.randint(1, 3)
            ability = ability_zscore[counter]
            ability = normal(ability_zscore, ability_zscore[counter])

            # create agents form data
            agent = SimClassAgent((x, y), self, agent_type, id[counter], behave[counter], behav2[counter],
                                  maths[counter],age[counter],fsm[counter], ability)
            # Place Agents on grid
            #x, y = self.grid.find_empty()
            self.grid.place_agent(agent, (x, y))
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
            agent_reporters={"x": lambda a: a.pos[0], "y": lambda a: a.pos[1], "id":"id","Inattentiveness_score": "behave",
                             "Hyber_Inattinteveness": "behave_2", "S_math": "s_math", "S_read": "s_read",
                             "E_math": "e_math", "E_read": "e_read", "ability": "ability",
                             "LearningTime": "countLearning", "disruptiveTend": "disruptiveTend", "Sigmoid":"Sigmoid", "neighbours":"neighbours"})

        self.running = True

    def step(self):


        self.learning = 0  # Reset counter of learing and disruptive agents
        self.distruptive = 0
        Daycounter = self.schedule.time
        self.daycounter+=1

        if self.daycounter == 45:
            self.schoolDay +=1
            self.daycounter = 0
        #self.schoolDay = self.schedule.steps
        #if self.schedule.steps == 45:
        #    self.schoolDay +=1
        self.datacollector.collect(self)
        self.schedule.step()
        Daycounter =+1

        # collect data
        self.datacollector.collect(self)

        if self.schedule.steps == 8550 or self.running == False:
            self.running = False
            dataAgent = self.datacollector.get_agent_vars_dataframe()
            dataAgent.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-117/all-Sigmoid-scallingFormula-3.csv')
            """""
            data = dataAgent
            data.sort_values(by='E_math')
            data = data.reset_index()  # make sure indexes pair with number of rows
            if self.NumberofGroups == 3:
                ThreeBreaks(data)
            else:
                TwoBreaks(data)


            #df['quantile'] = pd.qcut(df['e_maths'], q=3, labels=['group_1', 'group_2','group_3'])
            breaks = jenkspy.jenks_breaks(data['E_math'], nb_class=self.NumberofGroups)
            data['Classification'] = pd.cut(data['E_math'],
                        bins=breaks,
                        labels=[1, 2, 3],
                        include_lowest=True)

            Group1 = data[data ['Classification'] == 1]
            Group2 = data[data ['Classification'] == 2]
            Group3 = data[data ['Classification'] == 3]
            dataConvert = data.to_numpy()
            Group11 = Group1.to_numpy()
            Group22 = Group2.to_numpy()
            Group33 = Group3.to_numpy()
            print(data)
            FGroup = Group11[:,6]
            SGroup = Group22[:,6]
            TGroup = Group33[:,6]

            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)

            # Create the boxplot
             ## add patch_artist=True option to ax.boxplot()
            ## to get fill color
            bp = ax.boxplot((FGroup, SGroup, TGroup), patch_artist=True)
            ## change outline color, fill color and linewidth of the boxes
            for box in bp['boxes']:
             # change outline color
              box.set( color='#7570b3', linewidth=2)
              # change fill color
              box.set( facecolor = '#1b8f9e' )

            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
             whisker.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']:
              cap.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the medians
            for median in bp['medians']:
             median.set(color='#b2df8a', linewidth=2)

            ## change the style of fliers and their fill
            for flier in bp['fliers']:
              flier.set(marker='o', color='#e7298a', alpha=0.5)
            ax.set_xticklabels(['Group1', 'Group2', 'Group3'])

            ax.set_ylabel('End Math')
            #plt.show(bp)
            print("Classes are", Group11)
            dataConvert = dataAgent.to_numpy()
            plt.scatter(dataConvert[:,2], dataConvert[:,6],  s=50, alpha=0.5)
            print("Y is",dataConvert[:,6])
            plt.ylabel('Emath Score')
            plt.xlabel('Inattintiveness')
            #plt.show()
            dataAgent.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-115/all.csv')
            """
