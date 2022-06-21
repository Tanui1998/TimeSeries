import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# Import Custom functions


class AdvancedTimeSeriesModels:
    def __init__(self, data):
        self.data=data
        self.cols = list(self.data.columns)
        
        
    def visuals(self):
        """Plots all time series plots for all the variables
        """
        plt.style.use('dark_background')
        for i in range(len(self.cols)):
            df = self.cols[i]
            self.data[df].plot(figsize=(12, 5), color = "white") # Plotting the time series plots
            plt.title(df, size=14)
            plt.show()
        
        # fig, ax = plt.subplots(1, len(list(self.data.columns)), figsize=(16,3), sharex=True)
        # for col, i in dict(zip(self.data.columns, list(range(len(list(self.data.columns)))))).items():
        #     self.data[col].plot(ax=ax[i], legend=True, linewidth=1.0, color="red", sharex=True)     
        
        # fig.suptitle("Historical trends of levels variables", 
        #             fontsize=12, fontweight="bold")
            
    def Grangers_causality_test(self):
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table 
        are the P-Values. P-Values lesser than the significance level (0.05), implies 
        the Null Hypothesis that the coefficients of the corresponding past values is 
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        from Grangers_causality import grangers_causation_matrix
        self.maxlag=12
        df = grangers_causation_matrix(self.data,self.cols,self.maxlag)
        return df
    
    def cointegration_test(self):
        """Perform Johanson's Cointegration Test and Report Summary
    When two or more time series are cointegrated, 
    it means they have a long run, statistically significant relationship.
        """
        from cointegration_test import cointegration_test
        self.alpha=0.05
        cointegration_test(self.data,self.alpha)
        
    def train_test_split(self, n_obs=4):
        self.df_train = None
        self.df_test = None
        """Split data into train and test sets

        Args:
            n_obs (int, optional): This represnts the number of test obs. Defaults to 4.

        Returns:
            _type_: dataframe- returns the train and test datasets
        """
        from split_data import split_data
        self.n_obs = n_obs
        self.df_train, self.df_test = split_data(self.data,self.n_obs)
        return self.df_train, self.df_test
    
    def stationarity_check(self, df_train=None):
        """Perform Augmented Dickey Fuller Test on your dataset

        Args:
            df_train (the data to check stationarity on, optional): _description_. Defaults to None.
        """
        from stationaritycheck import adfuller_test
        self.df_train = df_train
        if self.df_train is not None:
            for name, column in self.df_train.iteritems():
                adfuller_test(column, name=column.name)
                print('\n')
        else:
            for name, column in self.data.iteritems():
                adfuller_test(column, name=column.name)
                print('\n')
                
    def remove_stationarity(self, df_train):
        self.df_train = df_train
        from differencing import differencing
        df_differenced = differencing(self.df_train)
        return df_differenced 
    
    def normality_check_for_vars(self,df_train=None):
        self.df_train = df_train
        from normality_check_for_vars import normality_check_vars
        if df_train is not None:
            normality_check_vars(self.df_train)
        else:
            normality_check_vars(self.data)
    
    def var_order_selection(self, df_differenced, maxlag):
        from varmodelorderselection import var_order_selection
        self.df_differenced = df_differenced
        self.maxlag=maxlag
        var_order_selection(self.df_differenced,self.maxlag)
        
    def VAR_model(self, df_differenced, order):
        self.df_differenced = df_differenced
        self.order = order
        from trainvarmodel import train_var_model
        model_fitted = train_var_model(self.df_differenced,self.order)
        print(model_fitted)
        
        # Get the lag order
        lag_order = model_fitted.k_ar
        # print(lag_order)  #> 4

        # Input data for forecasting
        forecast_input = df_differenced.values[-lag_order:]
        # Forecast
        fc = model_fitted.forecast(y=forecast_input, steps=self.n_obs)
        df_predicted = pd.DataFrame(fc, index=self.data.index[-self.n_obs:], columns=self.data.columns)
        print("Model's Prediction")
        return df_predicted
    
    def invert_transformation(self, df_train, df_predicted, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        self.df_train = df_train
        self.df_predicted = df_predicted
        self.second_diff = second_diff
        df_fc = df_predicted.copy()
        columns = df_train.columns
        for col in columns:        
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)].cumsum()
                # Roll back 1st Diff
                df_fc[str(col)+'_predicted'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
            else:
                df_fc[str(col)+'_predicted'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
        print("dataFrame of the inverted results")
        return df_fc
    
    def prediction_vs_actual(self,df_results_inverted):
        self.df_results_inverted = df_results_inverted
        fig, axes = plt.subplots(nrows=int(len(self.data.columns)/2), ncols=2, dpi=150, figsize=(10,10))
        for i, (col,ax) in enumerate(zip(self.data.columns, axes.flatten())):
            df_results_inverted[col+'_predicted'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
            self.df_test[col][-self.n_obs:].plot(legend=True, ax=ax);
            ax.set_title(col + ": prediction vs Actuals")
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

        plt.tight_layout();
        
    def prediction_vs_actual_for_stationary(self,df_prediction):
        self.df_prediction = df_prediction
        fig, axes = plt.subplots(nrows=int(len(self.data.columns)/2), ncols=2, dpi=150, figsize=(10,10))
        for i, (col,ax) in enumerate(zip(self.data.columns, axes.flatten())):
            df_prediction[col].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
            self.df_test[col][-self.n_obs:].plot(legend=True, ax=ax);
            ax.set_title(col + ": prediction vs Actuals")
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

        plt.tight_layout();
        
    
    def vecm_order_selection(self, df_train, maxlag):#freq_str, :
        self.df_train = df_train
        #self.freq_str = freq_str
        self.maxlag = maxlag
        from errorcorrectionmodel import ecm_order_selection
        ecm_order_selection(self.df_train, self.maxlag)#self.freq_str,
        
    def vecm_coint_rank_determination(self, df_train, lags):
        self.df_train = df_train
        self.lags = lags
        from errorcorrectionmodel import cointegration_rank_determination
        cointegration_rank_determination(self.df_train,self.lags)
        
    def vecm_model(self,df_train,lags,coint_rank):
        self.df_train = df_train
        self.lags=lags
        self.coint_rank = coint_rank
        from errorcorrectionmodel import error_correction_model
        vecm_fit = error_correction_model(self.df_train,self.lags,self.coint_rank)
        return vecm_fit
    
    def vecm_model_diagnostics(self, df_train, fitted_model, periods=15):
        self.df_train = df_train
        self.fitted_model = fitted_model
        self.periods = periods
        from errorcorrectionmodel import model_diagnostics
        model_diagnostics(self.df_train,self.fitted_model,self.periods)
        
    def vecm_prediction(self, df_test, fitted_model):
        self.df_test = df_test
        self.fitted_model = fitted_model
        from errorcorrectionmodel import prediction_from_model
        combined_df = prediction_from_model(self.df_test, self.fitted_model, self.n_obs)
        return combined_df