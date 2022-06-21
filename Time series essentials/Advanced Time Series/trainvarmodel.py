import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def train_var_model(df_differenced,order):
    model = VAR(df_differenced)
    model_fitted = model.fit(order)
    print(model_fitted.summary())
    import statsmodels.api as sm
    import sklearn.metrics as skm
    print("Model's R_square: " )
    print(skm.r2_score(model_fitted.fittedvalues+model_fitted.resid,
    model_fitted.fittedvalues))

    def check_serial_correlation():
        from statsmodels.stats.stattools import durbin_watson
        out = durbin_watson(model_fitted.resid)

        print("Serial Correlation Results: Durbin-Watson statistics")
        for col, val in zip(df_differenced.columns, out):
            print(col, ':', round(val, 2))
    check_serial_correlation()
    
    def normality_of_errors_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.

        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: The error terms are normally distributed')
        print()

        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(model_fitted.resid)[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    normality_of_errors_assumption()
    return model_fitted