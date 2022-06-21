from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.vector_ar.vecm import CointRankResults
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def ecm_order_selection(df_train,maxlags):#freq_str
        # import pandas as pd
        # VECM model fitting
        from statsmodels.tsa.vector_ar import vecm
        # pass "1min" frequency
        df_train.index = pd.DatetimeIndex(df_train.index)#.to_period(freq_str)
        model = vecm.select_order(df_train, maxlags=maxlags)
        print(model.summary())
   
def cointegration_rank_determination(df_train, lags):
        # cointegration rank determination
        from statsmodels.tsa.vector_ar.vecm import select_coint_rank
        print("if test statistic > critical values, the null of at most n cointegrating vectors are rejected.")
        print("if test statistic < critical values, the null of at most n cointegrating vectors cannot be rejected.")
        rank1 = select_coint_rank(df_train, det_order = 1, k_ar_diff = lags,
                                        method = 'trace', signif=0.01)
        print(rank1.summary())
        print("1st column in the table shows the rank which is the number of cointegrating relationships for the dataset, \nwhile the 2nd reports the number of equations in total.")
        rank2 = select_coint_rank(df_train, det_order = 1, k_ar_diff = lags, 
                              method = 'maxeig', signif=0.01)

        print(rank2.summary())

def error_correction_model(df_train,lags, coint_rank):
    # VECM fitting
    # VECM
    vecm = VECM(df_train, k_ar_diff=lags, coint_rank = coint_rank, deterministic='ci')
    """estimates the VECM on the data with n lags, n cointegrating relationship, and 
    a constant within the cointegration relationship"""
    vecm_fit = vecm.fit()
    print(vecm_fit.summary())
    return vecm_fit
    
    
def model_diagnostics(df_train, fitted_model, periods=15):
    print("R_square:")
    import statsmodels.api as sm
    import sklearn.metrics as skm
    print(skm.r2_score(fitted_model.fittedvalues+fitted_model.resid,
                 fitted_model.fittedvalues))
    print()
    print("Residual Autocorrelation:")
    # Residual auto-correlation
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(fitted_model.resid)
    for col, val in zip(df_train.columns, out):
        print((col), ':', round(val, 2))
        
    print()
    print("Normality Check:")
    from statsmodels.stats.diagnostic import normal_ad
    print('\n=======================================================================================')
    print('Assumption 2: The error terms are normally distributed')
    print()

    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(fitted_model.resid)[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Impulse-response plot
    print()
    print("Impulse-response plot")
    from statsmodels.tsa.vector_ar import irf
    irf = fitted_model.irf(periods)
    irf.plot(orth = False)
    plt.show()
    
    print()
    print("Impulse Response function")
    print()
    print("An IRF indicates what is the impact of an upward unanticipated one-unit change in the 'impulse' variable \non the 'response' variable over the next several periods (typically 10).")
    print()
    print("IRFs do not have coefficients. The original regressions as you specified them have the coefficients.")
    print()
    print("The IRFs has three main outputs: \nthe expected level of the shock in a given period surrounded by a 95% Confidence Interval (a low estimate and a high estimate).")
    print()
    cols = list(df_train.columns)
    for col in cols:
        print(f"Impulse Variable: {col}")
        plt.style.use('ggplot')
        irf.plot(impulse=col)
        plt.show()
        
def prediction_from_model(df_test,fitted_model,n_obs):
    # prediction
    pd.options.display.float_format = "{:.2f}".format
    forecast, lower, upper = fitted_model.predict(n_obs, 0.05)
    # print("lower bounds of confidence intervals:")
    # print(pd.DataFrame(lower.round(2)))
    # print("\npoint forecasts:")
    # print(pd.DataFrame(forecast.round(2)))
    # print("\nupper bounds of confidence intervals:")
    # print(pd.DataFrame(upper.round(2)))
    pd.options.display.float_format = "{:.2f}".format
    forecast = pd.DataFrame(forecast, index= df_test.index, columns= df_test.columns)
    cols = list(df_test.columns)
    for col in cols:
        forecast.rename(columns = {col:col+'_pred'}, inplace = True)
    
    combine = pd.concat([df_test, forecast], axis=1)
    pred = combine[sorted(combine.columns)]
    # def highlight_cols(s):
    #     color = 'yellow'
    #     return 'background-color: %s' % color

    # return pred.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['pred' in i for i in df_test.columns]])
    # print(forecast) 
    fig, axes = plt.subplots(nrows=int(len(df_test.columns)/2), ncols=2, dpi=150, figsize=(10,10))
    for i, (col,ax) in enumerate(zip(df_test.columns, axes.flatten())):
        pred[col+'_pred'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        df_test[col][-n_obs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": prediction vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout();
    # from sklearn.metrics import mean_absolute_error, mean_squared_error

    # for col in cols:
    #     mae = mean_absolute_error(pred[col], pred[col+'_pred'])
    #     mse = mean_squared_error(pred[col], pred[col+'_pred'])
    #     rmse = np.sqrt(mse)
    # sum = pd.DataFrame(index = ['Mean Absolute Error', 'Mean squared error', 'Root mean squared error'])
    # sum[f'Accuracy metrics :    {col.upper()}'] = [mae, mse, rmse]
    # print(sum)
    return pred