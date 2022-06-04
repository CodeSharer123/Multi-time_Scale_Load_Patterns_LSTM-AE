from data_drawer import draw_pie, draw_center_users, draw_center_users_barh, draw_center_users_barv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import sys
sys.path.append('..')

K = 4  # load K
user_info = pd.read_csv(BASIC_PATH)
user_info.set_index('CONS_NO', inplace=True)
new_datas = user_info.to_numpy()

km = KMeans(n_clusters=K)
labels = km.fit_predict(new_datas)
labels = np.array(labels)
km_cluster_centers = km.cluster_centers_

np.savetxt(os.path.join(SAVE_PATH, 'test5_3_3_cluster_centers.txt'),
           km_cluster_centers)
np.savetxt(os.path.join(
    SAVE_PATH, 'test5_3_3_labels.txt'), labels)

sys.path.append('..')

labels = np.loadtxt(LABEL_PATH)

labels = labels.astype(np.int32)

k = int(labels.max() + 1)

user_clusters = [[] for i in range(k)]

for i in range(user_info.shape[0]):
    user_clusters[labels[i]].append(user_info.iloc[i, :])

user_mean = []
nums_cluster = []

for i, user_cluster in enumerate(user_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(user_cluster))
    print('num: '+str(len(user_cluster)))
    user_cluster = np.array(user_cluster)
    user_mean.append(np.mean(user_cluster, axis=0))

print(user_mean)

draw_pie(nums_cluster, labels=['MMTP ' + str(i) for i in range(1, k + 1)],
         filepath=SAVE_PATH+'test5_3_4_pie', formats=('svg', 'png', 'tif'))

draw_center_users(4, 4, center_users=np.array(user_mean), ymax=None, label1='MMTP', label2='MLTP',
                  filepath=SAVE_PATH+'test5_3_4_user', formats=('svg', 'png', 'tif'))
draw_center_users_barv(np.array(user_mean), label1='MMTP', label2='MLTP',
                       filepath=SAVE_PATH+'test5_3_4_user_cz', formats=('svg', 'png', 'tif'))
draw_center_users_barh(np.array(user_mean), label1='MMTP', label2='MLTP',
                       filepath=SAVE_PATH+'test5_3_4_user_sp', formats=('svg', 'png', 'tif'))
