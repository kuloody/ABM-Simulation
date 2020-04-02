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

    #define the step function
    def step(self):
     noOfNeigh = 0
     red = 0
     green = 0
     yellow = 0
     for neighbor in self.model.grid.neighbor_iter(self.pos):
            noOfNeigh += 1
            if neighbor.type == 3:
                red += 1
            elif neighbor.type == 2:
                 yellow += 1
            else:
                 green+= 1
     agent_state = self.random.randrange(10)
     if self.model.Inattentiveness ==1 and self.model.quality> 3:
         if self.behave < 5:
             if self.type <= 2:
              self.type = 1
              self.model.learning += 1

     elif self.model.Inattentiveness ==0 and self.model.quality> agent_state and self.type == 2:
          self.type = 1
          self.model.learning += 1

     elif  self.model.control < agent_state and self.type == 1:
          self.type = 2

     elif self.model.hyper_Impulsive ==1 and self.model.control< agent_state and self.behave_2 > 5 and self.type == 2:
         self.type = 3
     elif self.model.hyper_Impulsive ==0 and self.model.control< agent_state and self.type == 2:
         self.type = 3
     elif self.model.control> agent_state and self.type == 3:
         self.type = 2
     elif red > 2 and self.type == 1:
         self.type = 2
     elif red > 5 and self.type == 2:
         self.type = 3

class Schelling(Model):
     '''
    Model class for the classroon model.
    '''

     def __init__(self, height=10, width=10, quality=1, Inattentiveness=0.0, control=3, hyper_Impulsive = 0):
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
        self.datacollector = DataCollector(
            {"learning": "learning"},  # Model-level count of learning agents
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
            agent_inattentiveness = self.random.randrange(10)
            agent_hyper_Impulsive = self.random.randrange(10)
            agent_type = 2
            agent = SchellingAgent((x, y), self, agent_type,agent_inattentiveness,agent_hyper_Impulsive)
            self.grid.position_agent(agent, (x, y))
            print('agent pos:',x,y)
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

     def step(self):
        '''
        Run one step of the model. If All agents are learning, halt the model.
        '''
        self.learning = 0  # Reset counter of learing agents
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        if self.learning == self.schedule.get_agent_count():
            self.running = False
