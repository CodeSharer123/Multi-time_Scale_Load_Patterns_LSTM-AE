import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

choices = ['LOW', 'BASE','HIGH']
debug = False

for choice in choices:
    SAVE_NAME = 'resident_DLPs_' + choice + '.csv'

    print('For', choice + ' to DLP')
    print('Debug is', debug)

    with open(BASIC_PATH+choice,  'rb') as f:
        datas = pickle.load(f)

    TIMES = ['01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00',
            '07:00:00', '08:00:00', '09:00:00', '10:00:00', '11:00:00', '12:00:00',
            '13:00:00', '14:00:00', '15:00:00', '16:00:00', '17:00:00', '18:00:00',
            '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00', '24:00:00']
    dates = []
    DLPs = []
    
    check_cnt = 0
    for df_data in datas:
        
        print(check_cnt)
        check_cnt += 1
        k = 0
        tmp_DLP = []
        tmp_date = []
        for j, row in df_data.iterrows():
            assert row['Date/Time'].split()[-1] != None
            if row['Date/Time'].split()[-1] == TIMES[k]:
                tmp_date.append(row['Date/Time'])
                if 'Electricity:Facility [kW](Hourly)' in df_data.columns:
                    tmp_DLP.append(row['Electricity:Facility [kW](Hourly)'])
                else:
                    tmp_DLP.append(row['Electricity:Facility [J](Hourly)'])
                k += 1
                if k == 24:
                    assert len(tmp_date) == 24
                    assert len(tmp_DLP) == 24
                    dates.append(tmp_date)
                    DLPs.append(tmp_DLP)
                    k = 0
                    tmp_DLP = []
                    tmp_date = []
            else:
                k = 0
                tmp_DLP = []
                tmp_date = []
                if row['Date/Time'].split()[-1] == TIMES[0]:
                    tmp_date.append(row['Date/Time'])
                    if 'Electricity:Facility [kW](Hourly)' in df_data.columns:
                        tmp_DLP.append(row['Electricity:Facility [kW](Hourly)'])
                    else:
                        tmp_DLP.append(row['Electricity:Facility [J](Hourly)'])
                    k += 1

    assert len(dates) == len(DLPs)

    day_list = []
    for i in range(len(dates)):
        print(i)
        day_list.append(dates[i][0].split()[0])
    DLP_df = pd.DataFrame(np.zeros((len(dates), 24)), index=day_list, columns=TIMES)
    for i, dlp in enumerate(DLPs):
        print(i)
        DLP_df.iloc[i, :] = np.array(dlp)
    
    for column in list(DLP_df.columns[DLP_df.isnull().sum() > 0]):
        mean_val = DLP_df[column].mean()
        DLP_df[column].fillna(mean_val, inplace=True)
    if debug:
        DLP_df.to_csv('./' + SAVE_NAME)
    else:
        DLP_df.to_csv(BASIC_PATH + SAVE_NAME)
