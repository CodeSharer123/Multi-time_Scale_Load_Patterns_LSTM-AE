import pandas as pd
import os

DLPs = pd.read_csv(BASIC_PATH)
DLPs.set_index('CONS_NO', inplace=True)
Users = list(set(DLPs.index))
Users = sorted(Users)
test = 0
for user in Users:
    DLPs.loc[user, :].to_csv(os.path.join(SAVE_PATH, str(user)+'.csv'))
    test += len(DLPs.loc[user, :])
