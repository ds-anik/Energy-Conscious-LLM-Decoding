import os
import pandas as pd

# Define input and output directories
#input_dir = "/home/alireza/D1/benchmarking_results/7B/code2text-java/contrastive-k25"  # Path to the folder with CSV files
#output_file = "/home/alireza/D1/benchmarking_results/7B/code2text-java/contrastive-k25/summary_results.csv"  # Path to the summary file 

input_dir = "/home/alireza/D1/benchmarking_results/test3/greedy_750"  # Path to the folder with CSV files
output_file = "/home/alireza/D1/benchmarking_results/test3/greedy_750/summary_results.csv"  # Path to the summary file 

# Initialize a list to store results
summary_data = []

# Process each CSV file in the directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_dir, file_name)

        # Load the DataFrame
        data = pd.read_csv(
            file_path,
            sep=",",
            header=0,
            parse_dates=['timestamp'],
            names=['timestamp', 'index', 'gpu_utilization [%]', 'power_draw [W]', 'clocks.applications.memory [MHz]', 'clocks.applications.graphics [MHz]']
        ) 

        # Preprocess data
        data = data.reset_index(drop=True)
        data.index = data.index + 2
        data["Active"] = data["gpu_utilization [%]"] > 0
        data["Group"] = (data["Active"] != data["Active"].shift()).cumsum()
        active_data = data[data["Active"]].copy()
        largest_group = active_data.groupby("Group").size().idxmax()
        largest_active_segment = active_data[active_data["Group"] == largest_group].copy()

        start_idx = largest_active_segment.index[0]
        end_idx = largest_active_segment.index[-1]
        start_idx_with_margin = max(start_idx - 0, 0)
        end_idx_with_margin = min(end_idx + 0, data.index[-1]) 

        largest_active_segment_with_margin = data.loc[start_idx_with_margin:end_idx_with_margin]
        largest_active_segment_with_margin = largest_active_segment_with_margin.drop(columns=["Active", "Group"])

        # Calculate results
        total_time = largest_active_segment_with_margin.shape[0]
        average_power = largest_active_segment_with_margin["power_draw [W]"].mean()
        total_energy = largest_active_segment_with_margin["power_draw [W]"].sum() 


        # Store results
        summary_data.append({
            "file_name": file_name,
            "total_time_seconds": total_time,
            "average_power_watts": round(average_power, 2),
            "total_energy_joules": round(total_energy, 2)
        })

# Save all results to a CSV file
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")       

