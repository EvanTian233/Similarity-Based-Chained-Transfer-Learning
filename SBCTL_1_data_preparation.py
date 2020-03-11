# Authors:	Yifang Tian
# Email:	ytian285@uwo.ca

# Step 1 in Similarity-Based Chained Transfer Learning algorithm
# Data cleaning
# Similarity set & Forecasting set preparation

# importing the needed libraries
import os
import glob
import pandas as pd
import numpy as np
from datetime import date
from workalendar.america import Ontario
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


###########################################################################################################################

# The dir of the files
# Need to take extra care about the dir of the folder while each step

os.chdir('/Users/farewell/Desktop/')

# The holiday calender
cal = Ontario()

# Pre - processing to 3 duplicated meter
'''
# loading in power data from dataset
data = pd.read_csv('121091.csv')

data['Date'] = pd.to_datetime(data['Date'])
# drop the duplicated rows
data.drop_duplicates(subset=None, keep='first', inplace=True)

print(data)

data.to_csv('121091.csv', sep=',', index=False)
'''


# Iterate through all the csv files in folder
csv_files = glob.glob('*.csv')
similarity_set = []

for filename in csv_files:

    data = pd.read_csv(filename)

    # Take the pre-fix of the file names
    filename = os.path.splitext(filename)[0]

    # Prepare data set
    '''
    # Drop useless columns
    #data = data.drop(['MTRID','Channel'],1)  # new set
    #data = data.drop(['P_MTRID','P_MTRCHN'],1) #old set

    # Converting 23:59 to 00:00
    #data['DT_TIMESTAMP'] = data['DT_TIMESTAMP'].dt.round('5min')
    '''


    # Visualize meters' usage profile
    '''
    data = data.loc[0:1344]
   # os.chdir('/Users/farewell/Desktop/Meters_new_dataset/See_new_meter/')
    print("\nSeeing new meter {} : ".format(filename))
    
    # daily sum
    #plt.figure(figsize=(33.60, 18.28))
    styles = ['b-', 'b-', 'b-']
    data_daily = data.groupby(['Date'])['Usage_W'].plot(style=styles,linewidth=1)

    #plt.title('Meter_{}_Daily_sum'.format(filename))
    #plt.savefig('Meter_{}_Daily_sum.png'.format(filename))
    '''

    # Visualize new meters' daily profile
    '''
    data['Date'] = pd.to_datetime(data['Date']) # Date/ DateTime
    data['Day_of_week'] = data['Date'].dt.weekday_name
    styles = ['b-', 'b-', 'r--', 'r--', 'b-', 'b-', 'b-']
    data.groupby(['Hour', 'Day_of_week'])['Usage_W'].mean().unstack().plot(style=styles,legend=False) # Usage_W/ P_usage
    #plt.title('Meter_{}_hourly_sum'.format(filename))

    lines = [Line2D([0], [0], color='b', linewidth=1.5, linestyle='-'),
             Line2D([0], [0], color='r', linewidth=1.5, linestyle='--'), ]
    labels = ['Weekday', 'Weekend']
    plt.legend(lines, labels)
    plt.ylabel('Power usage(kW/h)')
    plt.xlabel('Hour')

    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/New_meters_using')
    '''

    # Datetime index to new meters
    '''
    print("\nDatetime index to new meter {} : ".format(filename))
    data['DT_TIMESTAMP'] = pd.date_range(start='2017-08-01', periods=46752, freq='15min', closed='left')
    print(data.dtypes)

    data.set_index("DT_TIMESTAMP", inplace=True)
    data.index = pd.to_datetime(data.index)
    data = data.shift(1,freq='15min')
    data = data.drop(['Date','Hour','Minute'],1)
    print(data)

    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Ready_for_feature_generation/')
    data.to_csv('{}.csv'.format(filename), sep=',', index=True)
    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/New_meters_using/')
    '''

    # Sum up old meter time interval from 5 min into 15 min
    '''
    print("\nSumming up old meter {} into 15 min interval: ".format(filename))
    data['DT_TIMESTAMP'] = pd.to_datetime(data['DT_TIMESTAMP'])
    data.set_index("DT_TIMESTAMP", inplace=True)
    data = data.resample('15T').sum()
    data = data[1:]
    print(data)
    print(data.shape)

    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Ready_for_feature_generation/')
    data.to_csv('{}.csv'.format(filename), sep=',', index=True)
    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Old_meters')
    '''

    # All meter feature generation
    '''
    print("\nGenerating feature for meter {}: ".format(filename))
    data['DT_TIMESTAMP'] = pd.to_datetime(data['DT_TIMESTAMP'])
    data = data.sort_values(by='DT_TIMESTAMP')
    data['hour'] = data['DT_TIMESTAMP'].dt.hour
    data['minute'] = data['DT_TIMESTAMP'].dt.minute

    # make column for day of week: monday=0, sunday=6
    data['day_of_week'] = data['DT_TIMESTAMP'].dt.dayofweek

    # make column for day of weekend: 1 if yes, 0 if no
    data['weekend'] = 0
    data.loc[data.day_of_week == 5, 'weekend'] = 1
    data.loc[data.day_of_week == 6, 'weekend'] = 1

    # making column for day of month
    data['day_of_month'] = data['DT_TIMESTAMP'].dt.day

    # make column for day of year
    data['day_of_yr'] = data['DT_TIMESTAMP'].dt.dayofyear

    # make column for month of year: January=1, December=12
    data['month_of_yr'] = data['DT_TIMESTAMP'].dt.month

    # make column for year
    data['year'] = data['DT_TIMESTAMP'].dt.year

    # make column for season: winter=1, fall=4
    data['season'] = (data['DT_TIMESTAMP'].dt.month % 12 + 3) // 3

    # columns for output
    data = data[['DT_TIMESTAMP','year', 'month_of_yr', 'day_of_yr','day_of_month',
                       'day_of_week', 'weekend', 'hour', 'minute', 'season',  'P_USAGE']]

    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Pre_v2/')
    data.to_csv('{}.csv'.format(filename), sep=',', index=False)
    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Ready_for_feature_generation/')
    '''

    # Add holiday features
    '''
    
    print("\nAdding holiday features for meter {}: ".format(filename))
    data['DT_TIMESTAMP'] = pd.to_datetime(data['DT_TIMESTAMP'])

    data['holiday'] = 0
    # print cal.holidays(2017)
    # print cal.is_holiday(date(2017, 12, 25))
    # print cal.is_working_day(date(2017, 12, 25))
    # cal.add_working_days(date(2012, 12, 23), 5)  # 5 working days after Xmas

    # label holidays as '1'
    for i in range(len(data)):
        year = data.loc[i, 'year']
        month = data.loc[i, 'month_of_yr']
        day = data.loc[i, 'day_of_month']

        if cal.is_holiday(date(year, month, day)):
            data.at[i, 'holiday'] = 1

    data = data[['DT_TIMESTAMP', 'year', 'month_of_yr', 'day_of_yr',
                   'day_of_month', 'day_of_week', 'weekend', 'holiday', 'hour', 'minute', 'season', 'P_USAGE']]

    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Training_set/')
    data.to_csv('{}.csv'.format(filename), sep=',', index=False)
    print('File saved!')
    os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Pre_v2/')
    '''

    # Generate similarity set (Part 1/ need to work with Part 2 together)
    '''
    print("\nGenerating similarity set using meter set in 15 mins interval {}: ".format(filename))
    data['DT_TIMESTAMP'] = pd.to_datetime(data['DT_TIMESTAMP'])
    data = data[0:35040]
    data = data.drop(['DT_TIMESTAMP'], axis=1)
    data.columns = [filename]

    data = data.reset_index()
    similarity_set.append(data)
    '''

# Generate similarity set (Part 2/ need to work with Part 1 together)
'''
similarity_set = pd.concat(similarity_set,axis = 1)
similarity_set = similarity_set.drop(['index'], axis=1)

print('Similariy set:')
print(similarity_set)

# Export the similarity set
os.chdir('/Users/farewell/Desktop/Meters_new_dataset/Clustering_set/')
similarity_set.to_csv('similarity_calculation.csv', sep=',', index=False)
'''

# For the visualization
#plt.show()