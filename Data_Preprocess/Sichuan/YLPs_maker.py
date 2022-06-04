import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import datetime

debug = False

print('Debug is', debug)

filenames = os.listdir(BASIC_PATH)

YLPs = []
infos = []
for filename in filenames:
    DLPs = pd.read_csv(os.path.join(BASIC_PATH, filename))
    nowtime = datetime.datetime(2018, 1, 1)
    day_delta = datetime.timedelta(days=1)
    # print(nowtime.strftime('%Y-%m-%d'))

    DLPs_ptr = 0  
    tmp_YLP = []
    infos.append(filename[:-4])
    for t in range(365):
        now_time = nowtime.strftime('%Y-%m-%d')
        if DLPs_ptr < len(DLPs) and now_time == str(DLPs.iloc[DLPs_ptr, 1]).split()[0]:
            tmp_YLP.append(
                DLPs.iloc[DLPs_ptr, 2:].to_numpy().astype(np.float32))
            DLPs_ptr += 1
        else:
            tmp_YLP.append(np.array(96 * [np.nan]))
        nowtime = nowtime + day_delta
    assert len(tmp_YLP) == 365
    mean_DLP = np.nanmean(tmp_YLP, axis=0)
    assert len(mean_DLP) == 96
    for i in range(365):
        if np.isnan(tmp_YLP[i][0]):
            tmp_YLP[i] = mean_DLP
    tmp_YLP = np.array(tmp_YLP).reshape(-1)
    assert len(tmp_YLP) == (96 * 365)
    assert DLPs_ptr == len(DLPs)

    YLPs.append(tmp_YLP)
    if debug:
        break

assert len(infos) == len(YLPs)
re_df = pd.DataFrame(YLPs, index=infos)

re_df.to_csv(r'YLPs_18.csv')
