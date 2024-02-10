import pandas as pd
import numpy as np
import time
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask import send_file
from model import SimClass
import os

# Get the directory of the current script
project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)


# Initialize SimClass instance with initial data
initial_data = pd.read_csv('OldPIPS-SAMPLE.csv')
sim_instance = SimClass(initial_data)

def generateDataset():
    # Read the content of 'dataset.csv' and put it in a dataframe
    df = pd.read_csv('dataset.csv')

    # Replace '#NULL!' with NaN
    df = df.replace('#NULL!', np.nan)

    # Remove rows with NULL values
    df = df.dropna()

    # Convert relevant columns to numeric types
    numeric_cols = ['Inattentiveness', 'Hyperactivity', 'Impulsiveness']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Apply conditions for highly disruptive and moderately disruptive rows
    highly_disruptive = df[(df['Inattentiveness'] > 4) | (df['Hyperactivity'] > 3) | (df['Impulsiveness'] > 0)]
    moderately_disruptive = df[(df['Inattentiveness'] >= 3) | (df['Hyperactivity'] > 1)]

    # Concatenate the highly disruptive and moderately disruptive rows
    filtered_df = pd.concat([highly_disruptive, moderately_disruptive])

    # Set a seed for the random number generator based on the current time
    np.random.seed(int(time.time()))

    # Generate a new random subset of 30 rows from the filtered DataFrame
    random_subset = filtered_df.sample(n=30)

    # Write the output to a new file named 'generated.csv'
    random_subset.to_csv('generated.csv', index=False)


@app.route('/generate_dataset')
def generate_dataset():
    try:
        # Call the function to generate the dataset
        generateDataset()
        # Trigger a popup alert indicating success using JavaScript
        alert_script = "<script>alert('Dataset generated successfully!');</script>"
        df = pd.read_csv('generated.csv')
        # Update the data attribute of the SimClass instance
        sim_instance.data = df
        return alert_script
    except Exception as e:
        # Log the exception
        print(f"An error occurred: {str(e)}")
        # Return a JSON response to the client indicating failure
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    json_data = request.json  # Parse JSON data
    df = pd.DataFrame(json_data)  # Convert JSON data to DataFrame

    # Update the data attribute of the SimClass instance
    sim_instance.data = df

    # Process the DataFrame as needed
    print('Received DataFrame:', df)

    return 'DataFrame received successfully', 200


# Define a route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/show_results')
def show_results():
    try:
        # Return the HTML file to the client
        # Concatenate the project directory with the file path
        file_path = os.path.join(project_dir, 'results_visualizing.html')

        # Return the HTML file to the client
        return send_file(file_path)
    except Exception as e:
        # Log the exception
        print(f"An error occurred: {str(e)}")
        # Return a JSON response to the client indicating failure
        return jsonify({'success': False, 'error': str(e)}), 500
