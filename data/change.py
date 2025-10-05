import pandas as pd

# Load the CSV file
file_path = r'koi_predictions_with_scores.csv'
df = pd.read_csv(file_path)

# Update the 'kepler_name' column to ensure an underscore before the last index
def add_underscore_to_name(name):
    if '_' not in name:
        parts = name.split(' ')
        if len(parts) > 1:
            parts[-1] = f"_{parts[-1]}"
        return '-'.join(parts)
    return name

df['kepler_name'] = df['kepler_name'].apply(add_underscore_to_name)

# Save the updated CSV file
df.to_csv(file_path, index=False)
print("Planet names updated successfully!")