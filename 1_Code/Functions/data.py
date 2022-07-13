import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

def get_data_yfinance(_t):
    """
    1) Parameters
    ----------
    - _t : STRING
        TICKER.
    2) Returns
    -------
    - data : DATAFRAME
        Table with Open, High, Low, Close and Dividends.
    """
    data = yf.Ticker(_t).history(period="max")
    data = data[['Open','High','Low','Close','Dividends']]
    data.ffill(inplace = True)
    data = data[data.index <= datetime(2022, 6, 30)]
    return data

def dividends_norm_ret(series, series_dividends):
    """
    1) Parameters
    ----------
    - series : ARRAY
        Time series of Closing prices.
    - series_dividends : ARRAY
        Time series of Dividends.
    2) Returns
    -------
    DATAFRAME
        Dataframe of returns adjusted. It removes the return of the day
        with dividends and changes it by the real return. Logistic returns are
        used.
    """
    df_adj_ret = pd.DataFrame(columns = ['Close', 'Dividends', 'Returns',
                                        'Cl_Div','Adj_Ret'])
    df_adj_ret['Close'] = series
    df_adj_ret['Dividends'] = series_dividends
    df_adj_ret['Returns'] =  np.log(series).diff()
    df_adj_ret['Cl_Div'] = series + series_dividends.cumsum()
    df_adj_ret['Adj_Ret'] = np.log(df_adj_ret['Cl_Div'] / df_adj_ret['Cl_Div'].shift(1)) * \
        (df_adj_ret['Dividends'] != 0) + df_adj_ret['Returns'] * (df_adj_ret['Dividends'] == 0)
    return df_adj_ret