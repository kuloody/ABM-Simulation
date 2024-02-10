import os
import numpy as np
import pandas as pd
from mesa.batchrunner import BatchRunner
from model import SimClass

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
target_backend = 'cuda'

# Read the CSV file
# df = pd.read_csv('OldPIPS+Classroon_ID.csv')
df = pd.read_csv('/home/zsrj52/Downloads/SimClass/dataset/OldPIPS+Classroon_ID.csv')

# Drop the first column (assuming it's unnamed)
df.drop(df.columns[0], axis=1, inplace=True)

# Remove empty rows or null values
df.dropna(inplace=True)

# Group the dataset by the Classroom_ID
grouped_data = df.groupby('Classroom_ID')

# Create an empty DataFrame to hold aggregated data
aggregate_data = pd.DataFrame(columns=["Classroom_ID", "id", "Inattentiveness", "Hyperactivity", "Impulsiveness",
                                       "Start_maths", "Start_Reading", "End_maths", "End_Reading",
                                       "Start_Vocabulary", "fsm", "IDACI", "ability", "countLearning",
                                       "disruptiveTend", "Sigmoid", "neighbours"])

# Iterate over the groups and run a batch for each group
for classroom_id, data in grouped_data:
    # Define the fixed parameters for the model
    fixed_params = {
        "height": 10,
        "width": 10,
        "Seating": 1,
        "Nthreshold": 5,
        "data": data
    }

    variable_params = {"gamification_element": range(0, 6, 5)}
    # Create a batch runner for the SimClass model
    batch_run = BatchRunner(SimClass,
                            variable_params,
                            fixed_params,
                            iterations=1,
                            max_steps=8550,
                            agent_reporters={"id": "id",
                                             "Inattentiveness_score": "Inattentiveness",
                                             "Hyperactivity": "Hyperactivity",
                                             "Impulsiveness": "Impulsiveness",
                                             "Start_maths": "Start_maths",
                                             "Start_Reading": "Start_Reading",
                                             "End_maths": "End_maths",
                                             "End_Reading": "End_Reading",
                                             "Start_Vocabulary": "Start_Vocabulary",
                                             "FSM": "fsm",
                                             "IDACI": "IDACI",
                                             "ability": "ability",
                                             "LearningTime": "countLearning",
                                             "disruptiveTend": "disruptiveTend",
                                             "Sigmoid": "Sigmoid",
                                             "neighbours": "neighbours"})

    # Run the batch for this Classroom_ID
    batch_run.run_all()

    # Get and save the agent data to a CSV file for each classroom
    data_collector_agents = batch_run.get_agent_vars_dataframe()
    data_collector_agents.to_csv(f'Simulation-BatchRun-Gamification2-Classroom_{classroom_id}.csv', index=False)

    # Append the agent data to the aggregate DataFrame
    data_collector_agents["Classroom_ID"] = classroom_id
    aggregate_data = pd.concat([aggregate_data, data_collector_agents])

# Save the aggregated data to a single CSV file
aggregate_data.to_csv('/home/zsrj52/Downloads/SimClass/Simulations-120/Simulation-BatchRun-Gamification2.csv', index=False)
