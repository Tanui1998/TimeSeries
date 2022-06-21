import pandas as pd
import numpy as np

def differencing(data):
    df_differenced = data.diff().dropna()
    return df_differenced