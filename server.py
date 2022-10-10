from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import VisualizationElement
import numpy as np
from model import SimClass


class RightPanelElement(VisualizationElement):
    local_includes = ["RightPanelModule"]
    #js_code = "elements.push(new RightPanelModule());"
    def __init__(self):
        new_element = "new RightPanelModule()"
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        return f"""
<h4 style="margin-top:0">Model Variables</h4>
<table>
    <tr><td style="padding: 5px;">Learning Students الطلاب المتعلمون:</td><td style="padding: 5px;">{model.learning:.2f}</td></tr>
    <tr><td style="padding: 5px;">Disruptive Students الطلاب المشاغبون:</td><td style="padding: 5px;">{model.distruptive:.2f}</td></tr>
    <tr><td style="padding: 5px;">Current school day اليوم الدراسي الحالي</td><td style="padding: 5px;">{model.schoolDay}</td></tr>
</table>
"""

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
     agent_maths = [agent.s_math for agent in model.schedule.agents]
     ave = model.datacollector.get_model_vars_dataframe()
     ave.drop(columns=['disruptiveTend','Learning Students'])
     #x = sum(agent_maths)
     N = len(agent_maths)
     #B = x/N
     hist = np.histogram(ave, bins=self.bins)[0]
     return [int(x) for x in hist]


class TeacherMonitorElement(RightPanelElement):
    def __init__(self):
        pass

    def render(self, model):
        return f"""
        <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
<h4 style="margin-top:0">Model Variables</h4>
<table>
    <tr><td style="padding: 5px;" </td><p dir="rtl" lang="ar" style="color:#e0e0e0;font-size:20px;">رَبٍّ زِدْنٍي عِلمًا</p><td style="padding: 5px;">{model.learning:.2f}</td></tr>
    <tr><td style="padding: 5px;">Learning Students معلم:</td><td style="padding: 5px;">{model.learning:.2f}</td></tr>
    <tr><td style="padding: 5px;">Disruptive Students:</td><td style="padding: 5px;">{model.distruptive:.2f}</td></tr>
    <tr><td style="padding: 5px;">Current school day</td><td style="padding: 5px;">{model.schoolDay}</td></tr>
</table>
"""
class simElement(TextElement):


    def __init__(self):
        pass

    def render(self, model):
         agent_maths = [agent.s_math for agent in model.schedule.agents]
         ave = model.datacollector.get_model_vars_dataframe()
         ave = ave["Average End Math"]

         return "Learning Students: " + str(model.learning)


def simclass_draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "learning.png", "Layer": 0,  "text":agent.pos}
    type = agent.get_type()

    if type == 3:
        portrayal["Shape"] = "disruptive.png"
        #portrayal["Color"] = ["red", "red"]
        #portrayal["stroke_color"] = "#00FF00"

    if type == 2:
        portrayal["Shape"] = "passive.jpg"
        #portrayal["Color"] = ["yellow", "yellow"]
        #portrayal["stroke_color"] = "#00FF00"
    if type == 1:
        portrayal["Shape"] = "learning.png"
        #portrayal["Color"] = ["green", "green"]
        #portrayal["stroke_color"] = "#000000"

    return portrayal
def hist(model):
    Average = model.datacollector.get_model_vars_dataframe()
    Average.plot()

sim_element = simElement()
canvas_element = CanvasGrid(simclass_draw, 6, 5, 400, 400)
sim_chart = ChartModule([{"Label": "Learning Students", "Color": "green"},{"Label": "Distruptive Students", "Color": "red"},{"Label": "Average End Math", "Color": "black"}])
rightChart = RightPanelElement()
model_params = {
    "height": 5,
    "width": 6,
    "quality": UserSettableParameter("slider", "Teaching quality تحكم المعلم", 5.0 , 0.00, 5.0, 1.0),
    "control": UserSettableParameter("slider", "Teacher Control جودة التدريس", 5.0 , 0.00, 5.0, 1.0),
    "Inattentiveness": UserSettableParameter("slider", "Inattentiveness عدم الانتباه", 1.0 , 0.00, 1.0, 1.0),
    "hyper_Impulsive": UserSettableParameter("slider", "Hyperactivity فرط الحركة ", 1.0 , 0.00, 1.0, 1.0),
    "AttentionSpan": UserSettableParameter("slider", "Attention Span", 5.0 , 0.00, 5.0, 1.0),
     "Nthreshold": UserSettableParameter("slider", "Disrubtive Rang تأثير الجيران", 5.0 , 0.00, 5.0, 1.0),
     "NumberofGroups": UserSettableParameter('choice', 'Number of Groups عدد المجموعات',3,choices=[3,2])

}

histogram = HistogramModule(list(range(10)), 200, 500)
server = ModularServer(SimClass,
                       [rightChart,canvas_element, sim_element,sim_chart],
                       "SimClass", model_params)
