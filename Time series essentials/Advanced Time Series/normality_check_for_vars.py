import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normality_check_vars(df_train):
    from scipy import stats
    cols = list(df_train.columns)
    for col in cols:
        stat,p = stats.normaltest(df_train[col])
        print('Statistics=%.3f, p=%.3f' % (stat,p))
        alpha = 0.05
        if p > alpha:
            print(f'{col} Data looks Gaussian (fail to reject H0)')
            print(f'{col}: Kurtosis of normal distribution: {stats.kurtosis(df_train[col])}')
            print(f'{col}: Skewness of normal distribution: {stats.skew(df_train[col])}')
        else:
            print(f'{col} Data do not look Gaussian (reject H0)')
            print(f'{col}: Kurtosis of distribution: {stats.kurtosis(df_train[col])}')
            print(f'{col}: Skewness of distribution: {stats.skew(df_train[col])}')
        print('______________')