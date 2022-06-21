import pandas as pd

def split_data(data, n_obs):
    df_train, df_test = data[0:-n_obs], data[-n_obs:]
    return df_train, df_test