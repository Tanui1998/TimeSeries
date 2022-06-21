from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha): 
    """
    Perform Johanson's Cointegration Test and Report Summary
    When two or more time series are cointegrated, 
    it means they have a long run, statistically significant relationship.
    """
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)