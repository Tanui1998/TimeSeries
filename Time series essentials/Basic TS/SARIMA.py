# General
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import warnings

# stats
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Settings
plt.style.use("seaborn")
# Ignore harmless warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (15, 4)


# Necessary  function!!
def dataSplitter(data):
    """
    This method splits the data into train and test sets.
    The train test sets will follow the ratio provided,
    which by default is 90% for train test and 10% for the test set
    :return: data_train, data_test
    """
    ratio = 0.9
    # Calculating 90% of the data. User has the option to change on the fly
    size = int(len(data) * ratio)
    # Splitting the data
    data_train, data_test = data.iloc[:size], data.iloc[size:]
    return data_train, data_test


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.90)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.arima.ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# The classes start here
# First class: run_time_series()
class RunTimeSeries:
    """
    A class to aid in:
    -simple sarima model selection
    -time series plots
    -stationarity tests
    -Descriptive statistics
    """

    def __init__(self, data):
        self.data = data
        self.data_train = None
        self.data_test = None
        self.ratio = 0.9
        self.model = "additive"
        self.lags = 5
        self.upper_loop = 2
        self.period = None
        self.season = 12

    def descriptive_statistics(self):
        """
        Prints out the descriptive statistics
        :param data: The time series data which should be a data frame object
        :return: None
        """
        print("""------------------------
                -------------------------------
                Descriptive statistics""")
        print(self.data.describe())

    def time_all_plots(self):
        """
        Prints all time series plots required for analysis
        :param data: The time series data which should be a fully preprocessed data frame
        :param model: The model, either additive or multiplicative
        :param lags: The number of previous time points to consider for the plots
        :param period: required for the decomposition plots
        :return: None
        """
        print("""
        All Important time series plots
        """)
        self.data.plot(figsize=(20, 5), color="darkslategrey")  # Plotting the time series plots
        plt.title("Time Plot", size=10)
        plt.show()
        # Plotting the decomposition plots; trend, seasonality and randomness
        decomposition = seasonal_decompose(self.data,
                                           model=self.model,
                                           period=self.period)
        # decomposition.plot()
        seasonality = decomposition.seasonal
        trend = decomposition.trend
        seasonality.plot(color='darkslategrey')
        plt.title("Seasonality", size=10)
        plt.show()
        trend.plot(color='darkslategrey')
        plt.title("Trend", size=10)
        plt.show()
        resid = decomposition.resid
        resid.plot(color='darkslategrey')
        plt.title("Residuals", size=10)
        plt.show()
        # The ACF
        sgt.plot_acf(self.data, lags=self.lags,
                     #zero=False,
                     color="darkslategrey",
                     title="ACF")
        plt.show()
        # The PACF
        sgt.plot_pacf(self.data, alpha=0.05,
                      lags=self.lags, #zero=False,
                      method=("ols"),
                      color="darkslategrey",
                      title="PACF")
        plt.show()

    def augmented_dickey_fuller_test(self):
        """
        This a function to print out the results of the augmented dickey fuller test
        in a more appealing and understandable way
        :param data: The data to check for stationarity which must be a DataFrame
        :return: test-statistic, critical-values, p-value, lag, observations, maximized_criteria,
        :decides if the data is stationary or not.
        """
        print("""
        Stationarity Test - Checking whether our data is stationary
        Augmented Dickey Fuller test (ADF Test) is a common statistical test
        used to test whether a given Time series is stationary or not.
        It is one of the most commonly used statistical test when it comes
        to analyzing the stationarity of a series.
        The statsmodel package provides a reliable implementation of the ADF test 
        via the adfuller() function in statsmodels.tsa.stattools. It returns the following outputs:

        The p-value
        The value of the test statistic
        Number of lags considered for the test
        The critical value cutoffs.
        When the test statistic is lower than the critical value shown, 
        you reject the null hypothesis and infer that the time series is stationary
        """)

        p_value = sts.adfuller(self.data)[1]
        test_statistic = sts.adfuller(self.data)[0]
        lag = sts.adfuller(self.data)[2]
        observations = sts.adfuller(self.data)[3]
        critical_values = sts.adfuller(self.data)[4]
        maximized_criteria = sts.adfuller(self.data)[5]

        print("""   
        null-hypothesis: The data is non-stationary
        alternative-hypothesis: The data is stationary
        """)
        print(f"""  
        p-value: {p_value},
        test-statistic: {test_statistic},
        lag: {lag},
        observation: {observations},
        critical-values: {critical_values},
        maximized-criteria; {maximized_criteria}""")
        if test_statistic < critical_values['5%']:
            print(f"The data is stationary. The p-value is {round(p_value, 3)} thus we reject null")
        else:
            print(f"The data is non-stationary.The p-value is {round(p_value, 3)} thus we fail to reject null")

    def KPSS_test(self, **kw):
        """
        KPSS test is a statistical test to check for stationarity of a series around a deterministic trend.
        Like ADF test, the KPSS test is also commonly used to analyse the stationarity of a series.
        However, it has couple of key differences compared to the ADF test in function and in practical usage
        The KPSS test, short for, Kwiatkowski-Phillips-Schmidt-Shin (KPSS), is a type of Unit root test that tests
        for the stationarity of a given series around a deterministic trend.
        In other words, the test is somewhat similar in spirit with the ADF test. A common misconception,
        however, is that it can be used interchangeably with the ADF test.
        This can lead to misinterpretations about the stationarity,
        which can easily go undetected causing more problems down the line.
        In python, the statsmodel package provides a convenient implementation of the KPSS test.
        A key difference from ADF test is the null hypothesis of the KPSS test is that the series is stationary.
        So practically, the interpretaion of p-value is just the opposite to each other.
        That is, if p-value is < signif level (say 0.05), then the series is non-stationary.
        Whereas in ADF test, it would mean the tested series is stationary.
        kpss_test(series)
        Interpreting KPSS test:
        The output of the KPSS test contains 4 things:

        The KPSS statistic
        p-value
        Number of lags used by the test
        Critical values
        The p-value reported by the test is the probability score based on which you can decide
        whether to reject the null hypothesis or not. If the p-value is less than a predefined
        alpha level (typically 0.05), we reject the null hypothesis. The KPSS statistic is the actual test statistic
        that is computed while performing the test. For more information no the formula,
        the references mentioned at the end should help.
        In order to reject the null hypothesis, the test statistic should be greater than the provided critical values.
        If it is in fact higher than the target critical value, then that should automatically reflect in a low p-value.
        That is, if the p-value is less than 0.05, the kpss statistic will be greater than the 5% critical value.
        Finally, the number of lags reported is the number of lags of the series that was actually used
        by the model equation of the kpss test. By default, the statsmodels kpss() uses the ‘legacy’ method.
        By default, the statsmodels kpss() uses the ‘legacy’ method.
        In legacy method, int(12 * (n / 100)**(1 / 4)) number of lags is included, where n is the length of the series.
        To implement the KPSS test, we’ll use the kpss function from the statsmodel.
        The code below implements the test and prints out the returned outputs and interpretation from the result.
        # KPSS test
        from statsmodels.tsa.stattools import kpss
        def kpss_test(series, **kw):
            statistic, p_value, n_lags, critical_values = kpss(series, **kw)
            # Format Output
            print(f'KPSS Statistic: {statistic}')
            print(f'p-value: {p_value}')
            print(f'num lags: {n_lags}')
            print('Critial Values:')
            for key, value in critical_values.items():
                print(f'   {key} : {value}')
            print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
        In KPSS test, to turn ON the stationarity testing around a trend,
        you need to explicitly pass the regression='ct' parameter to the kpss.
        An example is shown below:
        kpss_test(data, regression='ct')
        KPSS Statistic: 0.11743798430485435
        p-value: 0.1
        num lags: 10
        Critial Values:
           10% : 0.119
           5% : 0.146
           2.5% : 0.176
           1% : 0.216
        Result: The series is stationary
        """
        # KPSS test
        from statsmodels.tsa.stattools import kpss
        print("H0: The time series data is stationary")
        print("H1:The time series data is non-stationary")
        statistic, p_value, n_lags, critical_values = kpss(self.data, **kw)
        # Format Output
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
        print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

    def split_data(self):
        """
        This method splits the data into train and test sets.
        The train test sets will follow the ratio provided,
        which by default is 90% for train test and 10% for the test set
        :return: data_train, data_test
        """
        # Calculating 90% of the data. User has the option to change on the fly
        self.size = int(len(self.data) * self.ratio)
        # Splitting the data
        self.data_train, self.data_test = self.data.iloc[:self.size], self.data.iloc[self.size:]
        return self.data_train, self.data_test

    def simple_sarima(self):
        """
        This is a function to make the SARIMA process easy and fast
        to come up with the optimum parameters to fit the best model
        :param b: The number to be include in the range during the
        grid search for the right number of parameters.
        param data: The data in which you fit the ARIMA model on
        param s : The seasonality of the SARIMA part of the model
        :return: it returns the several ARIMA models with their AIC and log likelihoods
        to aid in the choice of the best model.
        """
        print("""
            Choosing the best model from the range provided
            """)
        p = d = q = range(0, self.upper_loop)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], self.season) for x in list(itertools.product(p, d, q))]
        best_score, best_order, best_seas_order = float("inf"), None, None
        best_score_2nd, best_order_2nd, best_seas_order_2nd = float("inf"), None, None
        best_score_3rd, best_order_3rd, best_seas_order_3rd = float("inf"), None, None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.data_train,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    aic = results.aic
                    if aic < best_score:
                        best_score, best_order, best_seas_order = results.aic, param, param_seasonal

                    print(f'ARIMA{param}x{param_seasonal}season:{self.season} - AIC:{results.aic} - LLF:{results.llf}')
                except:
                    continue
        print('Best Order%s Seasonal_order%s AIC=%.3f' % (best_order, best_seas_order, best_score))


class ForecastTimeData:
    """
    ForecastTimeData class uses the data provided which must be ready for time series analysis
    and uses the best fitting model to produce forecasts.
    It also performs diagnostics on the results and returns the diagnostics plots
    """

    def __init__(self, data):
        self.data = data
        self.data_train = dataSplitter(data)[0]
        self.data_test = dataSplitter(data)[1]
        self.ratio = 0.9
        self.order = (0, 0, 0)
        self.seasonal_order = (1, 1, 1, 12)
        self.results = None
        self.lags = [10]
        self.results_resid = None
        self.start_date = None
        self.end_date = None
        self.steps = 10
        self.title = None
        self.y_label = None

    def best_results(self, order, seasonal_order):
        """
        This is a function to fit the best model based on aic and llf obtained from
        run_time_series function
        :param order: This is the ARIMA order that best suits the data. Entered as a tuple
        :param seasonal_order: The order of the seasonal part of the data. Entered as tuple
        :param enforce_stationarity: To be described
        :param enforce_invertibility:To be described
        :return: results
        """
        self.order = order
        self.seasonal_order = seasonal_order
        model = sm.tsa.statespace.SARIMAX(self.data_train,
                                          order=self.order,
                                          seasonal_order=self.seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

        self.results = model.fit()
        self.results_resid = self.results.resid
        print(self.results.summary().tables[1])
        return self.results, self.results_resid

    def results_diagnostics(self, results):
        """
        Plots the QQ-Plots and other diagnostic checks of the fitted model
        :param results: The best fitting model that has been obtained prior
        If The residual plots seem to lie around the mean of zero,
        following an almost normal distribution, and
        The errors also lie around the 45 degree line, our model is performing good.
        :return: None
        """
        self.results = results
        self.results.plot_diagnostics(figsize=(16, 8))
        plt.show()

    def ljung_test(self, results_resid):
        """
        Performs the ljung box test on our data
        The Ljung-Box test is a statistical test that checks if autocorrelation exists in a time series.
        It uses the following hypotheses:
        H0: The residuals are independently distributed.
        HA: The residuals are not independently distributed; they exhibit serial correlation.
        Ideally, we would like to fail to reject the null hypothesis.
        That is, we would like to see the p-value of the test be greater than 0.05
        because this means the residuals for our time series model are independent, which is often
        an assumption we make when creating a model.
        To perform the Ljung-Box test on a data series in Python,
        we can use the acorr_ljungbox() function from the statsmodels library
        which uses the following syntax:
        acorr_ljungbox(x, lags=None)
        where: x: The data series lags: Number of lags to test
        :param results_resid: results.resid
        :param lags:
        :param return_df:
        :return: None
        """
        print("""
            The Ljung-Box test is a statistical test that checks if autocorrelation exists in a time series 
            It uses the following hypotheses: H0: The residuals are independently distributed. 
            HA: The residuals are not independently distributed; they exhibit serial correlation. 
            Ideally, we would like to fail to reject the null hypothesis. 
            That is, we would like to see the p-value of the test be greater than 0.05 because this means 
            the residuals for our time series model are independent, 
            which is often an assumption we make when creating a model.
            """)
        self.results_resid = results_resid
        # perform Ljung-Box test on residuals with lag=10
        print(sm.stats.acorr_ljungbox(self.results_resid, lags=self.lags, return_df=True))

    def prediction_check(self, start_date, end_date):
        """
        Check the performance of our data by predicting
        already existing ones and comparing their patterns.
        Very crucial to know if we are on the right track.
        :param start_date: The date to start our prediction
        :param end_date: The date to end our prediction
        :return: df_pred
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_train, self.data_test = dataSplitter(self.data)
        df_pred = self.results.predict(start=self.start_date, end=self.end_date)
        # Visualizing the predictions
        # df_pred[self.start_date:self.end_date].plot(figsize=(20, 5), color="red")
        # self.data_test[self.start_date:self.end_date].plot(figsize=(20, 5), color="blue")
        plt.rcParams["figure.figsize"] = (20, 6)
        plt.plot(df_pred[self.start_date:self.end_date], label="prediction", color="red")
        plt.plot(self.data_test[self.start_date:self.end_date], label="actual", color="midnightblue")
        plt.xlabel("Date")
        plt.title("Predictions VS Actual", size=24)
        plt.legend(loc="upper right")
        plt.show()
        return df_pred[self.start_date:self.end_date]

    def forecasts_visualized(self, title, y_label):
        """
        This function produces forecasted data and visualizes it.
        :param steps: The steps to make forecasts
        :param ylabel: the labes for the y-axis
        :param title: The title of the visualization.
        :return: None
        """
        self.y_label = y_label
        self.title = title
        model = sm.tsa.statespace.SARIMAX(self.data,
                                          order=self.order,
                                          seasonal_order=self.seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit()
        pred_uc = results.get_forecast(steps=self.steps)
        pred_ci = pred_uc.conf_int()
        ax = self.data.plot(label='observed', figsize=(14, 7), color="darkslategrey")
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color="magenta")
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()
        plt.show()

    def show_forecast(self):
        """
        A method that outputs the forecasts of the time series process
        :return: None
        """
        print(self.results.get_forecast(steps=self.steps).summary_frame())

    def save_forecast(self):
        """
        Saving the predicted data to excel.
        :return: None
        """
        prediction = self.results.get_forecast(steps=self.steps).summary_frame()
        import os
        os.chdir(r"C:\Users\HP\Desktop\data")
        prediction.to_excel(input("Provide name ending with .xlsx: "))
