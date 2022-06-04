import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import datetime

debug = False

print('Debug is', debug)

filenames = os.listdir(BASIC_PATH)


isok = False
re_df = None
for filename in filenames:
    DLPs = pd.read_csv(os.path.join(BASIC_PATH, filename))
    nowtime = datetime.datetime(2018, 1, 1)
    day_delta = datetime.timedelta(days=1)
    # print(nowtime.strftime('%Y-%m-%d'))
    WLPs = []
    infos = []

    DLPs_ptr = 0  
    for t in range(52):
        tmp_WLP = []
        infos.append(filename[:-4] + '_' + nowtime.strftime('%Y-%m-%d'))
        for tt in range(7):
            now_time = nowtime.strftime('%Y-%m-%d')
            if DLPs_ptr < len(DLPs) and now_time == str(DLPs.iloc[DLPs_ptr, 1]).split()[0]:
                tmp_WLP.append(
                    DLPs.iloc[DLPs_ptr, 2:].to_numpy().astype(np.float32))
                DLPs_ptr += 1
            else:
                tmp_WLP.append(np.array(96 * [np.nan]))
            nowtime = nowtime + day_delta
        assert len(tmp_WLP) == 7
        mean_DLP = np.nanmean(tmp_WLP, axis=0)
        assert len(mean_DLP) == 96
        for i in range(7):
            if np.isnan(tmp_WLP[i][0]):
                tmp_WLP[i] = mean_DLP
        tmp_WLP = np.array(tmp_WLP).reshape(-1)
        tmp_WLP[np.isnan(tmp_WLP)] = 0
        assert len(tmp_WLP) == (96 * 7)
        WLPs.append(tmp_WLP)
    if DLPs.iloc[-1, 1].split()[0][5:] == '12-31':
        assert DLPs_ptr == len(DLPs) - 1
    else:
        assert DLPs_ptr == len(DLPs)
    assert len(infos) == len(WLPs)
    assert len(WLPs[0]) == (96 * 7)
    tmp_df = pd.DataFrame(np.array(WLPs))
    tmp_df['Info'] = infos
    tmp_df.set_index('Info', inplace=True)
    if debug:
        break
    tmp_df.to_csv(os.path.join(SAVE_PATH, filename))
    if not isok:
        isok = True
        re_df = tmp_df
    else:
        re_df = pd.concat([re_df, tmp_df], axis=0)

re_df.to_csv('WLPs_18.csv')


