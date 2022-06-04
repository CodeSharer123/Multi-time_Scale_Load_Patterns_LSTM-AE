import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

choices = ['LOW', 'BASE','HIGH']
debug = False


for choice in choices:
    SAVE_NAME = 'resident_YLPs_' + choice + '.csv'
    print('For', choice + ' to YLP')
    print('Debug is', debug)

    with open(BASIC_PATH + choice,  'rb') as f:
        datas = pickle.load(f)

    pure_datas = []

    for df_data in datas:
        if df_data.shape[0] != 8760:
            continue
        if 'Electricity:Facility [kW](Hourly)' in df_data.columns:
            pure_datas.append(df_data['Electricity:Facility [kW](Hourly)'].to_numpy())
        else:
            pure_datas.append(df_data['Electricity:Facility [J](Hourly)'])


    pure_datas = np.array(pure_datas)

    df_tmp = pd.DataFrame(pure_datas) 
    for column in list(df_tmp.columns[df_tmp.isnull().sum() > 0]):
        mean_val = df_tmp[column].mean()
        df_tmp[column].fillna(mean_val, inplace=True)

    if debug:
        df_tmp.to_csv('./' + SAVE_NAME)
    else:
        df_tmp.to_csv(BASIC_PATH + SAVE_NAME)