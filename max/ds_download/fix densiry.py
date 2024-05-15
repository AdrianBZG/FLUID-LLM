import pickle
import os

base_dir = './ds/MGN/airfoil_dataset'
mode = 'valid'

files = sorted(os.listdir(f'{base_dir}/{mode}'))
# Filter files that start with the specified string
filtered_files = [file for file in files if file.startswith("save")]

for file_name in filtered_files:
    print(file_name)

    with open(f"{base_dir}/{mode}/{file_name}", 'rb') as f:
        x = pickle.load(f)

    if 'densiry' in x.keys():
        x['density'] = x.pop('densiry')
    else:
        print("densiry already corrected")
        continue

    # Save modified file back
    with open(f"{base_dir}/{mode}/{file_name}", "wb") as f:
        pickle.dump(x, f)

