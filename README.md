# TimeSeriesFundamentals
All you need to know about time series
## Basic Time series 
- Contains ARIMA and SARIMA Model classes
- Plots the time series plots and ACF and PACF, as well as decomposition plots.

   ARIMA.time_all_plots()
  
   SARIMA.time_all_plots()
- Checks for stationarity of each variable Using both ADF and KPSS  tests

   ARIMA.augmented_dickey_fuller_test() 
   
   SARIMA.augmented_dickey_fuller_test() 
   
   ARIMA.KPSS_test()
   
   SARIMA.KPSS_test()
  
- Automates the selection of the best fitting ARIMA and SARIMA models.
- SARIMA model: (P,D,Q)*(p,d,q,s); with (P,D,Q) showing the non-seasonal part while (p,d,q,s) shows the seasonal part.
- The SARIMA class have the maximum orders in both the AR(P), Integration(D) and MA(Q) parts being 2 , but can be adjusted.
- The SARIMA class have the maximum seasonal orders in both the AR(p), Integration(d) and MA(q) parts being 2 and the maximum season being 12, but can be adjusted.

  SARIMA.simple_sarima()
