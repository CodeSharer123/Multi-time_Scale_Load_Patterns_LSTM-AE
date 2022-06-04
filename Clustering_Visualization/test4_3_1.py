import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import copy
from data_drawer import draw_LPs, draw_center_LPs, draw_pie
import os


K = 4
DLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'new_resident_DLPs_{choice}.csv')
    if DLPs is None:
        DLPs = Loads
    else:
        DLPs = pd.concat(
            [DLPs, Loads], axis=0)

Dates = DLPs['Date'].to_list()
CONS_NOs = []
for date in Dates:
    CONS_NOs.append(date.split('_')[1])
DLPs.drop(['Date'], axis=1, inplace=True)
DLPs = DLPs.to_numpy()


np.random.seed(233)
np.random.shuffle(DLPs)
np.random.seed(233)
np.random.shuffle(CONS_NOs)


labels = np.loadtxt(os.path.join(
    LABEL_PATH, 'test4_2_1_labels_'+str(K)+'.txt'))
labels = labels.astype(np.int32)

k = int(labels.max() + 1)
CONS_NOs_set = list(set(CONS_NOs))
columns = ['cluster'+str(i+1) for i in range(k)]
result_df = pd.DataFrame(
    np.zeros((len(CONS_NOs_set), k)), index=CONS_NOs_set, columns=columns)
assert len(CONS_NOs) == len(labels)

for i in range(len(labels)):
    result_df.loc[CONS_NOs[i], 'cluster' +
                  str(labels[i] + 1)] = result_df.loc[CONS_NOs[i], 'cluster' + str(labels[i] + 1)] + 1

result_df.to_csv(os.path.join(SAVE_PATH, 'user_info_'+str(K)+'.csv'))
