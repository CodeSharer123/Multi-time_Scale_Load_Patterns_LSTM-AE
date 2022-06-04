import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

debug = False
choices = ['LOW', 'BASE', 'HIGH']

for choice in choices:

    print('For', choice + ' to MLP')
    print('Debug is', debug)
    
    DLPs = pd.read_csv(BASIC_PATH+choice+'.csv')
    MLPs = []
    Days = []
    Months = []

    tmp_MLP = []
    tmp_day = []
    last_month = int(DLPs.iloc[0, 0].split('/')[0])
    for j, row in DLPs.iterrows():
        times = row['Date'].split('_')[0].split('/')
        month = int(times[0])
        day = int(times[1])
        if month == last_month:
            tmp_day.append(day)
            tmp_MLP.append(DLPs.iloc[int(j), 1:].to_numpy())
        else:
            Months.append(str(last_month)+'_'+row['Date'].split('_')[1])
            Days.append(tmp_day)
            MLPs.append(tmp_MLP)

            tmp_MLP = []
            tmp_day = []
            tmp_day.append(day) 
            tmp_MLP.append(DLPs.iloc[int(j), 1:].to_numpy())
            
        last_month = month
    
    Months.append(str(last_month)+'_'+row['Date'].split('_')[1]) 
    Days.append(tmp_day) 
    MLPs.append(tmp_MLP)
    
    assert len(Months) == len(Days) == len(MLPs)
    
    re_df = pd.DataFrame(np.zeros((len(Months), 31 * 24)),
                                             index=Months)
    for i in range(len(Months)):
        print(i)
        tmp_MLP = np.array(MLPs[i])
        add_MLP = tmp_MLP.mean(axis=0)
        for j in range(1, 32): 
            if j not in Days[i]:
                MLPs[i].insert(j-1, add_MLP)
        re_df.iloc[i, :] = np.array(MLPs[i]).reshape(-1)
    
    re_df.to_csv(SAVE_PATH)
