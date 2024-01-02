import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
target_backend='cuda'
import numpy as np
import pandas as pd

from model import SimClass
from mesa.batchrunner import BatchRunner
df = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS+Classroon_ID.csv')
df =df.dropna()
path2 = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS-SAMPLE.csv')
path3 = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/NewPIPS-SAMPLE.csv')

path = [path3,path2]
print(type(path3))
print('Here is path3',path3)
#print(path)
data_dict = dict(tuple(df.groupby('Classroom_ID')))

#new_dict = dict([(value, key) for key, value in pathArray.items()])

# pairs in the dictionary
result = data_dict.items()
print('Items of dict', type(result))
# Convert object to a list
data = list()
print(data[1:3])
print(type(data_dict.values()))
for item in data_dict.values():
   print(type(item[:]))
   data.append(tuple(item[:]))
#print(type(data))
#print(data[1:4])
# Convert list to an array
#numpyArray = np.array(data)
#dataFrame = pd.DataFrame.from_dict(pathArray)

KeysList = list(data_dict.keys())
#resultList = list(pathArray.items())
# printing the resultant list of a dictionary keys
#Test the tuples in batchrunner

Test_tuple = ('Hi I am a test Tuple item 1', 'Hi I am a test Tuple item 2')
Test_Dict = {0: ('Hi I am a test Tuple item 1'), 1:('Hi I am a test Tuple item 2')}
result = Test_Dict.items()

# Convert object to a list
#data = list(result)
fixed_params = {"width": 10,
               "height": 10,
                #"Inattentiveness": 1,
                #"hyper_Impulsive": 1,
                #"quality": 5,
                #"control": 5,
                #"AttentionSpan": 5
                "Seating":1,
                "Nthreshold": 5,
                "data":data_dict

                }
variable_params = { "key": KeysList}
#print(variable_params)
batch_run = BatchRunner(SimClass,
                        variable_params,
                        fixed_params,
                        iterations=1,
                        max_steps=8550,


                               # Model-level count of learning agents
                               # For testing purposes, agent's individual x and y
                       agent_reporters= {"id":"id","Inattentiveness_score": "Inattentiveness",
                             "Hyperactivity": "Hyperactivity","Impulsiveness":"Impulsiveness", "Start_maths": "Start_maths", "Start_Reading": "Start_Reading",
                             "End_maths": "End_maths", "End_Reading": "End_Reading","Start_Vocabulary":"Start_Vocabulary","FSM":"fsm","IDACI":"IDACI", "ability": "ability",
                             "LearningTime": "countLearning", "disruptiveTend": "disruptiveTend", "Sigmoid":"Sigmoid", "neighbours":"neighbours"})

batch_run.run_all()
data_collector_agents = batch_run.get_agent_vars_dataframe()
#data_collector_model = batch_run.get_collector_model()
data_collector_agents.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-120/Simulation-BtchRun-NthresholdTest5-15-allruns.csv')
