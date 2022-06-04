import pandas as pd
import os

filenames = os.listdir(filespaths)

re_df = pd.read_csv(os.path.join(filespaths, filenames[0]))
test = len(re_df)
for i in range(1, len(filenames)):
    tmp_df = pd.read_csv(os.path.join(filespaths, filenames[i]))
    re_df = pd.concat([re_df, tmp_df], axis=0)
    test += len(tmp_df)

assert test == len(re_df)

re_df = re_df.sort_values(by=["CONS_NO", "DATA_DATA"], ascending=[True, True])
re_df.set_index('CONS_NO', inplace=True)
re_df.to_csv(SAVE_PATH)
