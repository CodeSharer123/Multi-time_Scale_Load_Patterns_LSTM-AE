import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

debug = False

print('Debug is', debug)

filenames = os.listdir(BASIC_PATH)

isok = False
tmp_df = None
for i, filename in enumerate(filenames):
    print(i, filename)
    DLPs = pd.read_csv(os.path.join(BASIC_PATH, filename))

    MLPs = []
    Days = []
    Months = []
    tmp_MLP = []
    tmp_day = []
    last_month = int(DLPs.iloc[0, 1].split()[0].split('-')[1])
    for j, row in DLPs.iterrows():
        month = int(row['DATA_DATA'].split()[0].split('-')[1])
        day = int(row['DATA_DATA'].split()[0].split('-')[2])
        if month == last_month:
            tmp_day.append(day)
            tmp_MLP.append(DLPs.iloc[int(j), 2:].to_numpy())
        else:
            Months.append(filename[:-4]+'_'+str(last_month))
            Days.append(tmp_day)
            MLPs.append(tmp_MLP)
            tmp_MLP = []
            tmp_day = []
            tmp_day.append(day) 
            tmp_MLP.append(DLPs.iloc[int(j), 2:].to_numpy())

        last_month = month

    Months.append(filename[:-4]+'_'+str(last_month))
    Days.append(tmp_day)  
    MLPs.append(tmp_MLP)

    assert len(Months) == len(Days) == len(MLPs)

    tmp_df = pd.DataFrame(np.zeros((len(Months), 31 * 96)),
                          index=Months)
    for i in range(len(Months)):
        tmp_MLP = np.array(MLPs[i])
        add_MLP = tmp_MLP.mean(axis=0)
        for j in range(1, 32):  
            if j not in Days[i]:
                MLPs[i].insert(j-1, add_MLP)
        tmp_df.iloc[i, :] = np.array(MLPs[i]).reshape(-1)
    tmp_df['Info'] = Months
    tmp_df.set_index('Info', inplace=True)
    if debug:
        break
    tmp_df.to_csv(os.path.join(SAVE_PATH, filename))
    if not isok:
        isok = True
        re_df = tmp_df
    else:
        re_df = pd.concat([re_df, tmp_df], axis=0)
if not debug:
    re_df.to_csv(
        r'MLPs_18.csv')


