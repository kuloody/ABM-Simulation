import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from model import SimClass
from mesa.batchrunner import BatchRunner

fixed_params = {"width": 6,
               "height": 5,
                #"Inattentiveness": 1,
                #"hyper_Impulsive": 1,
                #"quality": 5,
                #"control": 5,
                #"AttentionSpan": 5
                }
variable_params = {"hyper_Impulsive": range(0, 2, 1), "Inattentiveness": range(0, 2, 1), "quality":range(0, 6, 1), "control":range(0, 6, 1) }

batch_run = BatchRunner(SimClass,
                        variable_params,
                        fixed_params,
                        iterations=3,
                        max_steps=8550,


                               # Model-level count of learning agents
                               # For testing purposes, agent's individual x and y
                       agent_reporters= { "id":"id","Inattentiveness": "Inattentiveness",
                       "Hyperactivity": "Hyperactivity", "Start_maths": "Start_maths", "End_maths": "End_maths", "ability": "ability",
                       "LearningTime": "countLearning","disruptiveTend":"disruptiveTend"})
batch_run.run_all()
data_collector_agents = batch_run.get_agent_vars_dataframe()
#data_collector_model = batch_run.get_collector_model()
data_collector_agents.to_csv(
                '/home/zsrj52/Downloads/SimClass/Simulations-118/Simulation-BtchRun.csv')
