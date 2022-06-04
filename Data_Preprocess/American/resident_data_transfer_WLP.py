
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

debug = False
choices = ['LOW', 'BASE', 'HIGH']
for choice in choices:
    print('For', choice + ' to WLP')
    print('Debug is', debug)

    DLPs = pd.read_csv(BASIC_PATH+choice+'.csv')

    WLPs = []
    Start_Dates = []


    tmp_WLP = []
    cnt = 0
    for j, row in DLPs.iterrows():
        if cnt == 0:
            Start_Dates.append(row['Date'])
        cnt += 1
        tmp_WLP.append(DLPs.iloc[int(j), 1:].to_numpy())
        if cnt == 7:
            assert len(tmp_WLP) == 7
            cnt = 0
            WLPs.append(np.array(tmp_WLP).reshape(-1))
            tmp_WLP = []

    if len(Start_Dates) == len(WLPs) + 1:
        Start_Dates = Start_Dates[0:-1]
    assert len(Start_Dates) == len(WLPs)
    assert len(WLPs[0]) == (24 * 7)
    re_df = pd.DataFrame(np.array(WLPs), index=Start_Dates)

    re_df.to_csv(SAVE_PATH)
