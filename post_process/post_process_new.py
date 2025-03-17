import pandas as pd

# Load the DataFrame
data = pd.read_csv(
    "/home/alireza/D1/benchmarking_results/7B/code2text-python/topp/gpu_monitor_topp_0.9_run_5.csv",
    sep=",",
    header=0,
    parse_dates=['timestamp'],
    names=['timestamp', 'index', 'gpu_utilization [%]', 'power_draw [W]']
) 

data = data.reset_index(drop=True)
data.index = data.index + 2 

# Mark active rows where GPUTL > 0
data["Active"] = data["gpu_utilization [%]"] > 0 

# Identify continuous blocks (segments of Active rows)
data["Group"] = (data["Active"] != data["Active"].shift()).cumsum()

# Filter active rows only
active_data = data[data["Active"]].copy()

# Find the largest continuous block of active rows
largest_group = (
    active_data.groupby("Group").size().idxmax()  # Get the group with the most rows
) 

# Filter rows belonging to the largest active group
largest_active_segment = active_data[active_data["Group"] == largest_group].copy()

# Add a margin of 1 row before and after the active segment
start_idx = largest_active_segment.index[0]  # First index of the largest active group
end_idx = largest_active_segment.index[-1]  # Last index of the largest active group

# Define start and end indices with the margin
start_idx_with_margin = max(start_idx - 0, 0)  # Ensure we don't go below index 0
end_idx_with_margin = min(end_idx + 0, data.index[-1])  # Ensure we don't exceed max index 

# Extract rows with the margin
largest_active_segment_with_margin = data.loc[start_idx_with_margin:end_idx_with_margin]

# Drop helper columns and reorder
largest_active_segment_with_margin = largest_active_segment_with_margin.drop(columns=["Active", "Group"])


# Calculate average power
average_power = largest_active_segment_with_margin["power_draw [W]"].mean() 

# Calculate total consumed energy (Joules)
total_energy = largest_active_segment_with_margin["power_draw [W]"].sum()  # Each row corresponds to 1 second

# Output results
print("Filtered DataFrame with Margin:")
print(largest_active_segment_with_margin)
print(f"\nTotal Time: {largest_active_segment_with_margin.shape[0]} seconds")
print(f"Average Power: {average_power:.2f} W")
print(f"Total Consumed Energy: {total_energy:.2f} Joules")  