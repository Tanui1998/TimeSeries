import statsmodels.api as sm
from SARIMA import RunTimeSeries, ForecastTimeData, dataSplitter, evaluate_arima_model
class Arima(RunTimeSeries, ForecastTimeData):
    def __init__(self, data=None, p_values=None, d_values=None, q_values=None):
        RunTimeSeries.__init__(self, data)
        ForecastTimeData.__init__(self, data)
        self.data = data
        self.p_values = p_values
        self.d_values = d_values
        self.q_values = q_values

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_arima_models(self):
        """
        Example:
        p_values = [0, 1, 2, 4, 6, 8, 10]
        d_values = range(0, 3)
        q_values = range(0, 3)
        evaluate_arima_models()
        :return: series of ARIMA models
        """
        dataset = self.data.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in self.p_values:
            for d in self.d_values:
                for q in self.q_values:
                    order = (p, d, q)
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.6f' % (order, mse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.6f' % (best_cfg, best_score))

    def best_arima_results(self, order):
        """
        This is a function to fit the best model based on aic and llf obtained from
        run_time_series function
        :param order: This is the ARIMA order that best suits the data. Entered as a tuple
        :param enforce_stationarity: To be described
        :param enforce_invertibility:To be described
        :return: results
        """
        train, test = dataSplitter(self.data)
        self.order = order
        model = sm.tsa.arima.ARIMA(train, order=self.order)

        self.results = model.fit()
        print(self.results.summary().tables[1])
        return self.results, self.results.resid