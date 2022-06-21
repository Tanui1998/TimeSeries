import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def var_order_selection(df_differenced, maxlag):
    """Selecting the best order for the VAR model

    Args:
        df_differenced (dataFrame): _A non-stationary dataset
        maxlag (_type_): _description_
    """
    model = VAR(df_differenced)
    for i in range(maxlag):
        result = model.fit(i)
        print('Lag Order =', i+1)
        print('AIC : ', result.aic)
        print('BIC : ', result.bic)
        print('FPE : ', result.fpe)
        print('HQIC: ', result.hqic, '\n')
    print()
    print("Compare with the one below")
    x = model.select_order(maxlags=maxlag)
    print(x.summary())

    