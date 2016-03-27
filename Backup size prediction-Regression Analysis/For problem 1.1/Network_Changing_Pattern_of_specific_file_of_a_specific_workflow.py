# -*- coding: utf-8 -*-
#This script is to see the changing patterns of different workflows
import numpy as np
import csv
import pylab as pl

#load data
with open('network_backup_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    data_list = [row[0:7] for row in reader]
    # all the data in np.array format
    data = np.array(data_list)
    
    data[data == 'Monday'] = 1
    data[data == 'Tuesday'] = 2
    data[data == 'Wednesday'] = 3
    data[data == 'Thursday'] = 4
    data[data == 'Friday'] = 5
    data[data == 'Saturday'] = 6
    data[data == 'Sunday'] = 7
    data[data == 'work_flow_0'] = 0
    data[data == 'work_flow_1'] = 1
    data[data == 'work_flow_2'] = 2
    data[data == 'work_flow_3'] = 3
    data[data == 'work_flow_4'] = 4
    data[data == 'File_0'] = 0
    data[data == 'File_1'] = 1
    data[data == 'File_2'] = 2
    data[data == 'File_3'] = 3
    data[data == 'File_4'] = 4
    data[data == 'File_5'] = 5
    data[data == 'File_6'] = 6
    data[data == 'File_7'] = 7
    data[data == 'File_8'] = 8
    data[data == 'File_9'] = 9
    data[data == 'File_10'] = 10
    data[data == 'File_11'] = 11
    data[data == 'File_12'] = 12
    data[data == 'File_13'] = 13
    data[data == 'File_14'] = 14
    data[data == 'File_15'] = 15
    data[data == 'File_16'] = 16
    data[data == 'File_17'] = 17
    data[data == 'File_18'] = 18
    data[data == 'File_19'] = 19
    data[data == 'File_20'] = 20
    data[data == 'File_21'] = 21
    data[data == 'File_22'] = 22
    data[data == 'File_23'] = 23
    data[data == 'File_24'] = 24
    data[data == 'File_25'] = 25
    data[data == 'File_26'] = 26
    data[data == 'File_27'] = 27
    data[data == 'File_28'] = 28
    data[data == 'File_29'] = 29


#changing pattern of a specific file in a workflow  
workflow = data[data[:,3] == '0'] # define workflow 0,1,2,3,4
workflow_file = workflow[workflow[:,4] == '0'] # define file 1,2,...29
workflow_file = workflow_file[0:120]

week = workflow_file[:,0].astype(np.float)
day = workflow_file[:,1].astype(np.float)
hour = workflow_file[:,2].astype(np.float)
x = (week-1)*168+(day-1)*24+hour
y = workflow_file[:,5].astype(np.float)

fig,ax = pl.subplots()
pl.plot(x, y)    
pl.title('Workflow0 File0') #change workflow name and filename
ax.set_xlabel('Time (h)')
ax.set_ylabel('Size of Backup (GB)')
pl.show()