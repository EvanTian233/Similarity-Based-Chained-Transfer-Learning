# Authors:	Yifang Tian
# Email:	ytian285@uwo.ca

# Step 2 in Similarity-Based Chained Transfer Learning algorithm
# Similarity calculation with similarity set


# importing the needed libraries
import os
import glob
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.spatial import distance

###########################################################################################################################



# Read similarity set
os.chdir('/Users/farewell/Desktop/')
data = pd.read_csv('similarity_calculation.csv')
x = data

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled = preprocessing.StandardScaler().fit_transform(x)

x_scaled = pd.DataFrame(x_scaled)
x_scaled.columns = x.columns
x_scaled = x_scaled.T

# Calculating Euclidean matrix between all possible source-target meter pairs
E_score = scipy.spatial.distance.cdist(x_scaled.iloc[:,1:], x_scaled.iloc[:,1:], metric='euclidean')
E_score = pd.DataFrame(E_score)
E_score.index = x_scaled.index
E_score.columns = x_scaled.index
print(E_score)
data = E_score.replace('0', np.nan)

# Preparation for the calculation
data = data.stack()
data.index.set_names(['model_1', 'model_2'], inplace=True)
data = data.drop_duplicates( keep='first', inplace=False)
data = data.reset_index()
data['model_2'] = pd.to_numeric(data['model_2'])
#data = data.set_index(['model_1', 'model_2'])
data.columns = ['model_1', 'model_2','similarity']
data[['model_1', 'model_2']] = data[['model_1', 'model_2']].astype(str)
print("The data would be like:")
print(data)

# Get the transfer learning path based on Euclidean distance
def get_transfer_path (data,meter_pair_set):

    # Get the min value
    if len(meter_pair_set) is 0:
        min = data['similarity'].min()
    else:
        # Get the index with the meters in set
        data_min = data.isin(meter_pair_set)

        # Filter data using the meter set
        data_min_index = data[data_min.values]
        min = data_min_index['similarity'].min()

    print('min:')
    print(min)
    id_list = data.loc[data['similarity'] == min]

    if len(data) > 0:
        model_1 = data[data.similarity == min].model_1.item()
        model_2 = data[data.similarity == min].model_2.item()

        # Add into meter pair set
        meter_pair_set.add(model_1)
        meter_pair_set.add(model_2)

        # Drop the min one according to the pair set
        data_to_drop = (data.model_1.isin(meter_pair_set)) & (data.model_2.isin(meter_pair_set))
        data = data[~data_to_drop]

        # Recursive
        sub_id_list = get_transfer_path(data,meter_pair_set)
        id_list = id_list.append(sub_id_list)
    return id_list

# Output transfer learning path
t_path = get_transfer_path(data,meter_pair_set=set())
print("the transfer learning path:")
print(t_path)
sort_set = set()
for i, row in t_path.iterrows():
    r_v_1 = row["model_1"]
    r_v_2 = row["model_2"]
    if r_v_2 in sort_set:
        t_path.at[i, 'model_1']  = r_v_2
        t_path.at[i, 'model_2']  = r_v_1

    sort_set.add(row["model_1"])
    sort_set.add(row["model_2"])

# Output the transfer learning path
print("the transfer learning path:")
print(t_path)
t_path.to_csv('SBCTL_path.csv', sep=',', index=False)