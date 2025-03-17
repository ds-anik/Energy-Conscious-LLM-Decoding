import csv
import json
from collections import defaultdict

# Input CSV file and output JSON file
csv_file = "./7B/code2text-java/contrastive-k25/summary_results.csv"  # Replace with your CSV filename
json_file = "./7B/code2text-java/contrastive-k25/energy_data.json"

# Initialize a dictionary to store the data in the desired format
data = defaultdict(lambda: defaultdict(list))

# Read the CSV file
with open(csv_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Extract data from the file_name column
        file_name = row["file_name"]
        energy = float(row["total_energy_joules"]) 

        # Extract decoding strategy, hyperparameter, and run info
        parts = file_name.split("_")
        #strategy = parts[2]
        strategy = "contrastive-k25"
        hyperparameter = parts[3] 
        #hyperparameter = f"({parts[3]},{parts[5]})"

        # Append the energy value to the corresponding list
        data[strategy][hyperparameter].append(energy)

# Convert defaultdict to a regular dict for saving as JSON
output_data = {k: dict(v) for k, v in data.items()}

# Save the data to a JSON file
with open(json_file, "w") as file:
    json.dump(output_data, file, indent=4)

print(f"Data has been saved to {json_file}")
