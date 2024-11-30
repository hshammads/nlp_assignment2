import os
import json
import requests
import pandas as pd
import numpy as np
from collections import defaultdict

# list of dates to fix
fix_dates = [
    '20241029', '20241030', '20241031', '20241101', '20241102', '20241103', '20241104', '20241105', '20241112',
]

# function to fix the JSON file
def fix_json_file(json_path, date):
    with open(json_path, 'r') as file:
        data = file.read()

    # Fix the invalid object separation by replacing '},{"' with ',{'
    fixed_data = data.replace('},{', ',')

    try:
        # Attempt to load the fixed data
        json_data = json.loads(fixed_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} for {date}")
        return None  

    # Save the fixed JSON data back to the file
    with open(json_path, 'w') as file:
        json.dump(json_data, file, indent=4)

    print(f"JSON file '{json_path}' has been fixed.")
    return json_data




class Helper:

    def retrieve_articles():
        data_folder = '..\\nlp_assignment2\\data'
        all_data = defaultdict(lambda: defaultdict(list))  # Initialize a defaultdict to store the data

        for date_folder in os.listdir(path=data_folder):
            date_path = os.path.join(data_folder, date_folder)

            if os.path.isdir(date_path):
                json_path = os.path.join(date_path, "news_data.json")

                # check if the date is in the fix_dates list
                if date_folder in fix_dates:
                    data = fix_json_file(json_path, date_folder)
                    print(json.dumps(data, indent=4))
                    
                    
        return all_data

if __name__ == "__main__":
    data = Helper.retrieve_articles()

