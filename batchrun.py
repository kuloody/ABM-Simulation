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
variable_params = { "quality":range(0, 6, 5)}
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
data_collector_agents.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-120/Simulation-BtchRun-Gamification2.csv')
