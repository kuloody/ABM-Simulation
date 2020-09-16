import random
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


def compute_ave(model):
    agent_maths = [agent.e_math for agent in model.schedule.agents]
    x = sum(agent_maths)
    N = len(agent_maths)
    B = x / N
    print('the AVARAGE', B, agent_maths)
    return B


def compute_ave_disruptive(model):
    agent_disruptiveTend = [agent.disruptiveTend for agent in model.schedule.agents]
    print('Calculate disrubtive tend original', agent_disruptiveTend)
    B = statistics.mean(agent_disruptiveTend)
    print('Calculate disrubtive tend after mean', agent_disruptiveTend)
    print('the AVARAGE', B, agent_disruptiveTend)
    return B


def compute_zscore(model, x):
    agent_behave = [agent.behave for agent in model.schedule.agents]
    print('Calculate variance', agent_behave)
    SD = stdev(agent_behave)
    mean = statistics.mean(agent_behave)
    zscore = (x - mean) / SD
    return zscore


def normal(agent_ability, x):
    # agent_ability = [agent.ability for agent in model.schedule.agents]
    minValue = min(agent_ability)
    maxValue = max(agent_ability)
    rescale = (x - minValue) / maxValue - minValue
    # We want to test rescaling into a different range [1,20]
    a = 1
    b = 2
    rescale = ((b - a) * (x - minValue) / (maxValue - minValue)) + a
    return rescale


class SimClassAgent(Agent):
    # 1 Initialization
    def __init__(self, pos, model, agent_type, behave, behave_2, math, ability):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.behave = behave
        self.behave_2 = behave_2
        self.s_math = math
        self.e_math = math
        self.ability = ability
        self.agent_state = self.random.randrange(6)
        self.greenState = 0
        self.redState = 0
        self.yellowState = 0
        self.disrubted = 0
        self.countLearning = 0
        self.disruptiveTend = behave

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

    # define the step function
    def step(self):
        # if self.type == 3:
        #    self.model.distruptive += 1
        #   self.disrubted += 1
        # self.changeState()
        if self.greenStateCange() == 1:
            self.changeState()
            self.set_disruptive_tend()
            self.agent_state = self.random.randrange(6)
            print('Hi change $$$$')
            return
        elif self.redStateCange() == 1:
            self.changeState()
            self.set_disruptive_tend()
            self.agent_state = self.random.randrange(6)
            print('Hi change $$$$')
            return
        elif self.yellowStateCange() == 1:
            self.changeState()
            self.set_disruptive_tend()
            self.agent_state = self.random.randrange(6)
            print('Hi change $$$$')
            return

        # self.yellowStateCange()
        # self.greenStateCange()
        # self.redStateCange()
        # self.changeState()
        # self.changeState()
        # self.set_disruptive_tend()
        print('agent state', self.agent_state)
        print('start math', self.s_math)
        print('end math', self.e_math)
        print('ability', self.ability)
        print('Learn Count', self.countLearning)
        # self.agent_state = self.random.randrange(6)
        # self.changeState()
        print('agent type', self.type)
        print('green state counter', self.greenState)
        # if self.type == 1:
        # change the state of the student in every step
        self.agent_state = self.random.randrange(6)

    # self.neighbourState()

    def redStateCange(self):
        count, red, yellow, green = self.neighbour()

        if self.disruptiveTend > compute_ave_disruptive(self.model) and (
                self.model.quality or self.model.control) <= self.agent_state:
            if self.type == 1:
                self.model.learning -= 1
            # if self.model.schedule.steps > 0 and self.yellowState >=3:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

        if self.model.hyper_Impulsive == 1 and self.model.control <= self.agent_state and self.behave_2 > 3:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

        if red > 5 and self.type == 2:
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return 1

    """
        if self.model.hyper_Impulsive == 0 and self.model.control < self.agent_state:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 3
            self.model.distruptive += 1
            self.disrubted += 1
            self.redState += 1
            self.yellowState = 0
            self.greenState = 0
            return
      """

    def yellowStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.disruptiveTend >= compute_ave_disruptive(
                self.model) and self.model.quality <= self.agent_state and self.behave <= 5:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 2
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.disruptiveTend >= compute_ave_disruptive(
                self.model) and self.model.control <= self.agent_state and self.behave_2 <= 3:
            if self.type == 1:
                self.model.learning -= 1
            self.type = 2
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.model.quality > self.agent_state and self.behave >= 5:
            self.type = 2
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        # At random if control is less than student state
        if self.model.control <= self.agent_state and self.type == 1:
            self.type = 2
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.model.learning -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return
        # At general if control is high turn into passive
        if self.model.control > self.agent_state and self.behave_2 > 3:
            self.type = 2
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            return 1

        if red > 2 and self.type == 1:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1

        # Change state based on majority of neighbours' color and agent's current color state

    def greenStateCange(self):

        count, red, yellow, green = self.neighbour()

        if self.disruptiveTend > compute_ave_disruptive(
                self.model) and self.model.quality > self.agent_state and self.behave < 5:
            if self.type <= 2:
                self.type = 1
                self.model.learning += 1
                self.set_start_math()
                self.redState = 0
                self.yellowState = 0
                self.greenState += 1
                return 1
        # this needs revision
        if self.disruptiveTend <= compute_ave_disruptive(self.model) and (
                self.model.quality or self.model.control) > self.agent_state and self.type <= 2:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return 1

        if self.model.hyper_Impulsive == 1 and self.model.control > self.agent_state and self.behave_2 <= 3:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return 1
        if self.disruptiveTend <= compute_ave_disruptive(self.model) and self.model.control > self.agent_state:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            return 1

        if green > 5 and self.type == 2:
            self.type = 1
            self.model.learning += 1
            self.set_start_math()
            return 1

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

        # Change to red if inattentiveness score is high and teaching quality is low and state is passive for long

        # Change to red if hyber impulsive score is high and teaching quality is low and state is passive for long

        # Change to attentive (green) is hyber impulsive score is low and teaching quality is high and state is passive for long
        if (self.model.quality or self.model.control) > 3 and self.yellowState >= 3:

            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1
        # Change to green if passive for long

        # Change to passive (yellow) if inattentiveness score is high and teaching control is low and state is green for long
        if self.model.control < 3 and self.redState > 4:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        # Similar to above but red fpr long
        # ##Change to passive (yellow) if inattentiveness score is high and teaching quality is low and state is green for long
        if self.model.quality < 3 and self.redState > 4:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        # Change to passive (yellow) if inattentiveness score is high and teaching control is high and state is green for long
        # Student will lose interest if inattentiveness score is high regardless of teaching quality
        if self.behave > 5 and self.model.quality > 3 and self.redState > 3:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        # Change to passive (yellow) if hyber impulsive score is high and teaching control is high and state is green for long
        # Student will lose focus if hyber impulsive score is high regardless of teaching control
        if self.behave_2 > 3 and self.model.control > 3 and self.redState > 3:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1

        # Change to yellow if inattentiveness score is low
        if self.model.control > 3 and self.redState > 2:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        # Change to yellow if hyper impulsive score is low
        if self.behave_2 <= 3 and self.model.quality < 3 and self.redState > 2:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        if self.behave_2 <= 3 and self.model.control <= 3 and self.redState > 2:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1
        if self.behave < 5 and self.model.quality < 3 and self.redState > 2:
            self.type = 2
            self.model.distruptive -= 1
            # self.disrubted += 1
            # self.redState += 1
            self.yellowState += 1
            self.greenState = 0
            return 1
        if self.yellowState > 5:
            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1
        if self.redState > 2 and self.disruptiveTend <= compute_ave_disruptive(self.model):
            self.type = 1
            if self.model.distruptive > 0:
                self.model.distruptive -= 1
            self.redState = 0
            self.yellowState = 0
            self.greenState += 1
            self.model.learning += 1
            self.set_start_math()
            return 1
        if self.greenState > self.model.AttentionSpan:
            self.type = 2
            self.redState = 0
            self.yellowState += 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
            return 1

    def set_start_math(self):
        #Increment the learning counter
        self.countLearning += 1
        if self.model.schedule.steps == 0:
            self.model.schedule.steps = 1
       # Scale Smath before using it to calculate end math score
        Scaled_Smath = (2.303 ** self.s_math) ** (1 / 5.975)
        total_learn = self.countLearning + Scaled_Smath
        self.e_math = (5.985 * math.log(total_learn) + (self.ability * random.normalvariate(1, 2)))


    # self.e_math = ((self.countLearning / self.model.schedule.steps) * (
    #      self.ability + random.normalvariate(2, 3))) + (self.s_math)

    def get_type(self):
        return self.type

    def set_disruptive_tend(self):

        self.initialDisrubtiveTend = compute_zscore(self.model, self.behave)

        print("HERE AFTER Z SCORE", self.initialDisrubtiveTend)
        if self.model.schedule.steps == 0:
            self.model.schedule.steps = 1

        self.disruptiveTend = (((self.disrubted / self.model.schedule.steps) - (
                self.countLearning / self.model.schedule.steps)) + self.initialDisrubtiveTend)



class SimClass(Model):
    '''
    Model class for the classroon model.
    '''

    def __init__(self, height=6, width=6, quality=1, Inattentiveness=0, control=3, hyper_Impulsive=0, AttentionSpan=0):
        '''
        '''

        self.height = height
        self.width = width
        self.quality = quality
        self.Inattentiveness = Inattentiveness
        self.control = control
        self.hyper_Impulsive = hyper_Impulsive
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(width, height, torus=True)
        self.AttentionSpan = AttentionSpan

        self.learning = 0
        self.distruptive = 0
        self.redState = 0
        self.yellowState = 0
        self.greenState = 0

        # agent_start_math = random.randrange(0,70,size=(36))
        agent_start_math = np.random.uniform(low=0, high=70, size=(36,))
        print('agent_start_math', agent_start_math)
        ability_zscore = stats.zscore(agent_start_math)
        print('ability_zscore', ability_zscore)
        # what kind of code will it be?
        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        # create all agent with passive state and generate a random score for inattentiveness
        counter = 0
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            # Control random behavior score of agents
            if Inattentiveness == 1:
                Inattentiveness_list = [1, 2, 3, 4] * 30 + [5, 6, 7] * 60 + [8, 9] * 10
            else:
                Inattentiveness_list = [1, 2, 3, 4] * 70 + [5, 6, 7, 8, 9] * 30
            agent_inattentiveness = self.random.choice(Inattentiveness_list)
            if hyper_Impulsive == 1:
                hyper_Impulsive_list = [1, 2, 3] * 30 + [4, 5, 6] * 70
            else:
                hyper_Impulsive_list = [1, 2, 3] * 70 + [4, 5, 6] * 30
            agent_hyper_Impulsive = self.random.choice(hyper_Impulsive_list)

            # Initial State for all student is passive
            agent_type = 2

            ability = normal(ability_zscore, ability_zscore[counter])
            if ability_zscore[counter] < 0:
                smath = self.random.randint(0, 17)
            else:
                smath = self.random.randint(17, 70)

            # Create Agents
            agent = SimClassAgent((x, y), self, agent_type, agent_inattentiveness, agent_hyper_Impulsive,
                                  smath, ability)
            # Place Agents on grid
            self.grid.position_agent(agent, (x, y))
            print('agent pos:', x, y)
            self.schedule.add(agent)
            counter += 1

        # Collect chosen data while running the model
        self.datacollector = DataCollector(
            {"distruptive": "distruptive",
             "learning": "learning",
             "Average": compute_ave,
             "disruptiveTend": compute_ave_disruptive},
            # Model-level count of learning agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1], "Inattentiveness": "behave",
             "Hyber_Inattinteveness": "behave_2", "S_math": "s_math", "E_math": "e_math", "ability": "ability",
             "LearningTime": "countLearning", "DisrubtiveTend": "disruptiveTend"})

        self.running = True

    def step(self):

        '''
        Run one step of the model. If All agents are learning, halt the model.
        '''

        self.learning = 0  # Reset counter of learing agents
        self.distruptive = 0
        self.datacollector.collect(self)
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)
        if self.schedule.steps == 12000.0 or self.running == False:
            self.running = False
            dataAgent = self.datacollector.get_agent_vars_dataframe()
            dataAgent.to_csv(
                '/home/zsrj52/Downloads/SimClass/Simulations19-08-2020/Simulation61-LowControl.csv')
