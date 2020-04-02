from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from model import Schelling


class HappyElement(TextElement):
    '''
    Display a text count of how many learning agents there are.
    '''

    def __init__(self):
        pass

    def render(self, model):
        return "Learning agents: " + str(model.learning)+" Distruptive agents: " + str(model.distruptive)


def schelling_draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}

    if agent.type == 3:
        portrayal["Color"] = ["red", "red"]
        portrayal["stroke_color"] = "#00FF00"
    if agent.type == 2:
        portrayal["Color"] = ["yellow", "yellow"]
        portrayal["stroke_color"] = "#00FF00"
    if agent.type == 1:
        portrayal["Color"] = ["green", "green"]
        portrayal["stroke_color"] = "#000000"
    return portrayal


happy_element = HappyElement()
canvas_element = CanvasGrid(schelling_draw, 6, 5, 400, 400)
happy_chart = ChartModule([{"Label": "learning", "Color": "green"},{"Label": "distruptive", "Color": "red"}])

model_params = {
    "height": 5,
    "width": 6,
    "quality": UserSettableParameter("slider", "Teaching quality", 5.0 , 0.00, 5.0, 1.0),
    "control": UserSettableParameter("slider", "Control", 5.0 , 0.00, 5.0, 1.0),
    "Inattentiveness": UserSettableParameter("slider", "Inattentiveness", 1.0 , 0.00, 1.0, 1.0),
    "hyper_Impulsive": UserSettableParameter("slider", "hyper_Impulsive", 1.0 , 0.00, 1.0, 1.0)
}

server = ModularServer(Schelling,
                       [canvas_element, happy_element, happy_chart],
                       "SimClass", model_params)
