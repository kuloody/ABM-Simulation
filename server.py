from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import VisualizationElement
import pandas as pd
import numpy as np
from model import SimClass
from panel_template import RightPanelElement

class HistogramModule(VisualizationElement):
    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        agent_maths = [agent.Start_maths for agent in model.schedule.agents]
        ave = model.datacollector.get_model_vars_dataframe()
        ave.drop(columns=['disruptiveTend', 'Learning Students'])
        # x = sum(agent_maths)
        N = len(agent_maths)
        # B = x/N
        hist = np.histogram(ave, bins=self.bins)[0]
        return [int(x) for x in hist]


class chartStyling(TextElement):
    def __init__(self):
        pass

    def render(self):
        return """ 
        <style>
            body {
                background-color: #FFFFFF; /* Replace with your desired background color */
            }
        </style>
        """


class simElement(TextElement):

    def __init__(self):
        pass

    def render(self, model):
        return """ 
        <style>
            body {
                background-color: #CCCCCC; /* Grey background color */
            }

        </style>
        """


def simclass_draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "learning.jpg", "Layer": 0, "text": agent.pos}
    type = agent.get_type()

    if type == 3:
        portrayal["Shape"] = "disruptive.jpg"
        # portrayal["Color"] = ["red", "red"]
        # portrayal["stroke_color"] = "#00FF00"

    if type == 2:
        portrayal["Shape"] = "passive.png"
        # portrayal["Color"] = ["yellow", "yellow"]
        # portrayal["stroke_color"] = "#00FF00"
    if type == 1:
        portrayal["Shape"] = "learning.jpg"
        # portrayal["Color"] = ["green", "green"]
        # portrayal["stroke_color"] = "#000000"

    return portrayal


def hist(model):
    Average = model.datacollector.get_model_vars_dataframe()
    Average.plot()


sim_element = simElement()
canvas_element = CanvasGrid(simclass_draw, 10, 10, 600, 600)

sim_chart = ChartModule(
    [{"Label": "Learning Students", "Color": "green"}, {"Label": "Distruptive Students", "Color": "red"},
     {"Label": "Average End Math", "Color": "black"}])

rightChart = RightPanelElement()

# Initialize SimClass instance with initial data
initial_data = pd.read_csv('OldPIPS-SAMPLE.csv')
sim_instance = SimClass(initial_data)
model_params = {
    "height": 10,
    "width": 10,
    "gamification_element": UserSettableParameter("slider", "gamification element", 5.0, 0.00, 5.0, 1.0),
    "teacher_level": UserSettableParameter("slider", "Teacher Level", 5.0, 0.00, 5.0, 1.0),
    "Seating": UserSettableParameter("slider", "Change Seats Every Lesson ", 1.0, 0.00, 1.0, 1.0),
    "State_Minutes": UserSettableParameter("slider", "Minutes of Change State ", 5.0, 1.00, 5.0, 1.0),
    "Inattentiveness": UserSettableParameter("slider", "Inattentiveness ", 1.0, 0.00, 1.0, 1.0),
    "hyper_Impulsive": UserSettableParameter("slider", "Hyperactivity   ", 1.0, 0.00, 1.0, 1.0),
    "AttentionSpan": UserSettableParameter("slider", "Attention Span", 5.0, 0.00, 5.0, 1.0),
    "Nthreshold": UserSettableParameter("slider", "Disruptive Range  ", 10.0, 5.0, 15.0, 1.0),
    "NumberofGroups": UserSettableParameter('choice', 'Number of Groups  ', 3, choices=[3, 2]),
    "data": sim_instance.data,  # Pass the initial data
}

histogram = HistogramModule(list(range(10)), 200, 500)

server = ModularServer(SimClass,
                       [rightChart, sim_element, canvas_element, sim_chart, chartStyling],
                       "SimClass", model_params)
