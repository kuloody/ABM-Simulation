from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector


class SchellingAgent(Agent):
    # 1 Initialization
    def __init__(self, pos, model, agent_type, behave, behave_2):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.behave = behave
        self.behave_2 = behave_2
        self.agent_state = self.random.randrange(6)
        self.redState = 0
        self.yellowState = 0
        self.greenState = 0

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
        if self.type == 3:
            self.model.distruptive += 1

        self.redStateCange()
        self.yellowStateCange()
        self.greenStateCange()
        self.changeState()
        print('agent type',self.type)
       # self.neighbourState()

    def redStateCange(self):
        count, red, yellow, green = self.neighbour()
        if self.model.Inattentiveness == 1 and self.model.quality <= self.agent_state and self.behave > 5:
            self.type = 3
            self.model.distruptive += 1
            self.redState =+1
            self.yellowState = 0
            self.greenState = 0

        if self.model.hyper_Impulsive == 1 and self.model.control < self.agent_state and self.behave_2 > 5:
            self.type = 3
            self.model.distruptive += 1
            self.redState =+1
            self.yellowState = 0
            self.greenState = 0

        if self.model.hyper_Impulsive == 1 and self.model.control > self.agent_state and self.behave_2 < 5 and self.type == 3:
            self.type = 3
            self.model.distruptive += 1
            self.redState =+1
            self.yellowState = 0
            self.greenState = 0

        if self.model.hyper_Impulsive == 0 and self.model.control < self.agent_state:
            self.type = 3
            self.model.distruptive += 1
            self.redState =+1
            self.yellowState = 0
            self.greenState = 0

        if red > 5 and self.type == 2:
            self.type = 3
            self.model.distruptive += 1
            self.redState =+1
            self.yellowState = 0
            self.greenState = 0

    def yellowStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.model.Inattentiveness == 1 and self.model.quality <= self.agent_state and self.behave < 5:
            self.type = 2
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0

        if self.model.Inattentiveness == 0 and self.model.quality > self.agent_state and self.behave >= 5 and self.type == 3:
            self.type = 2
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0

        if self.model.control < self.agent_state and self.type == 1:
            self.type = 2
            self.model.learning -= 1
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0

        if self.model.control > self.agent_state and self.type == 3:
            self.type = 2
            self.model.distruptive -= 1
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0

        if red > 2 and self.type == 1:
            self.type = 2
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0
            if self.model.learning > 0:
                self.model.learning -= 1
        Pturn_red = red/count
       # Pturn_green = self.model.control + self.model.quality
        Pturn_green = green/count
        Pturn_yellow = yellow/count

        if self.type == 3:
              Pturn_red += 0.2
        elif self.type == 2:
              Pturn_yellow += 0.2
        else:
                 Pturn_green+= 0.2
        colour = max(Pturn_red,Pturn_green,Pturn_yellow)
        if Pturn_yellow == colour:
            self.type = 2
            self.redState = 0
            self.yellowState =+ 1
            self.greenState = 0
            print('here yellow color')

    def greenStateCange(self):

        count, red, yellow, green = self.neighbour()
        if self.model.Inattentiveness == 1 and self.model.quality > self.agent_state and self.behave < 5:
            if self.type <= 2:
                self.type = 1
                self.model.learning += 1
                self.redState = 0
                self.yellowState = 0
                self.greenState =+ 1

        if self.model.Inattentiveness == 1 and self.model.quality > self.agent_state and self.behave < 5:
            if self.type <= 2:
                self.type = 1
                self.model.learning += 1
                self.redState = 0
                self.yellowState = 0
                self.greenState =+ 1

        if self.model.Inattentiveness == 0 and self.model.quality > self.agent_state and self.behave >= 5 and self.type <= 2:
            self.type = 1
            self.model.learning += 1
            self.redState = 0
            self.yellowState = 0
            self.greenState =+ 1

        if self.model.hyper_Impulsive == 1 and self.model.control > self.agent_state and self.behave_2 < 5 and self.type <= 2:
            if self.type <= 2:
                self.type = 1
                self.model.learning += 1
                self.redState = 0
                self.yellowState = 0
                self.greenState =+ 1

        if green > 5 and self.type == 2:
            self.type = 1
            self.model.learning += 1
        Pturn_red = red/count
       # Pturn_green = self.model.control + self.model.quality
        Pturn_green = green/count
        Pturn_yellow = yellow/count

        if self.type == 3:
              Pturn_red += 0.2
        elif self.type == 2:
              Pturn_yellow += 0.2
        else:
                 Pturn_green+= 0.2
        colour = max(Pturn_red,Pturn_green,Pturn_yellow)
        if Pturn_green == colour:
            self.type = 1
            self.model.learning += 1
            self.redState = 0
            self.yellowState = 0
            self.greenState =+ 1
            print('here green color')
    def neighbourState(self):
        count, red, yellow, green = self.neighbour()
        # calculate the probability of each colour
        Pturn_red = red/count
       # Pturn_green = self.model.control + self.model.quality
        Pturn_green = green/count
        Pturn_yellow = yellow/count

        if self.type == 3:
              Pturn_red += 0.2
        elif self.type == 2:
              Pturn_yellow += 0.2
        else:
                 Pturn_green+= 0.2
        colour = max(Pturn_red,Pturn_green,Pturn_yellow)
        if Pturn_red == colour:
            self.type = 3
        if Pturn_yellow == colour:
            self.type = 2
        if Pturn_green == colour:
            self.type = 1
            self.model.learning += 1
    def changeState(self):
        if self.behave > 5 and self.model.control < 3 and self.yellowState > 2:
            self.type = 3
        if self.behave_2 > 5 and self.model.quality < 3 and self.yellowState > 2:
            self.type = 3
        if self.behave > 5 and self.model.control < 3 and self.greenState > 2:
            self.type = 2
        if self.behave_2 > 5 and self.model.quality < 3 and self.greenState > 2:
            self.type = 2
        if self.behave > 5 and self.model.control > 3 and self.greenState > 5:
            self.type = 2
        if self.behave_2 > 5 and self.model.quality > 3 and self.greenState > 5:
            self.type = 2
    def get_type(self):
        return self.type
class Schelling(Model):
    '''
    Model class for the classroon model.
    '''

    def __init__(self, height=6, width=6, quality=1, Inattentiveness=0, control=3, hyper_Impulsive=0):
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

        self.learning = 0
        self.distruptive = 0
        self.datacollector = DataCollector(
            {"distruptive": "distruptive",
             "learning": "learning"},
            # Model-level count of learning agents
            # For testing purposes, agent's individual x and y
            {"x": lambda a: a.pos[0], "y": lambda a: a.pos[1]})

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        # create all agent with passive state and generate a random score for inattintiveness
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            # Control radome behavior of agents
            if Inattentiveness == 1:
                Inattentiveness_list = [1, 2, 3, 4] * 40 + [5, 6, 7, 8, 9] * 70
            else:
                Inattentiveness_list = [1, 2, 3, 4] * 70 + [5, 6, 7, 8, 9] * 40
            agent_inattentiveness = self.random.choice(Inattentiveness_list)
            if hyper_Impulsive == 1:
                hyper_Impulsive_list = [1, 2, 3, 4] * 30 + [5, 6, 7, 8, 9] * 70
            else:
                hyper_Impulsive_list = [1, 2, 3, 4] * 70 + [5, 6, 7, 8, 9] * 30
            agent_hyper_Impulsive = self.random.choice(hyper_Impulsive_list)
            # agent_hyper_Impulsive = self.random.randrange(10)
            agent_type = 2
            agent = SchellingAgent((x, y), self, agent_type, agent_inattentiveness, agent_hyper_Impulsive)
            self.grid.position_agent(agent, (x, y))
            print('agent pos:', x, y)
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model. If All agents are learning, halt the model.
        '''
        self.learning = 0  # Reset counter of learing agents
        self.distruptive = 0
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)
        if self.schedule.steps == 40.0:
            self.running = False

        if self.learning == self.schedule.get_agent_count():
            self.running = False
        if self.distruptive == self.schedule.get_agent_count():
            self.running = False
