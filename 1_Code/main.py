"""----------------------------------------------------------------------------
First, the  libraries
----------------------------------------------------------------------------"""
import pickle
import tensorflow as tf
import yfinance as yf
import scipy.optimize as sco
import pandas as pd
import numpy as np
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, MaxPooling3D, AveragePooling3D, ConvLSTM2D, Reshape, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import itertools
import pickle

from Functions import data as data_retrieve
from Functions import model_inputs
from Functions import sharpe_functions

"""----------------------------------------------------------------------------
Information download
----------------------------------------------------------------------------"""
_tickers = ['IBE.MC', 'ITX.MC', 'SAN.MC', 'TEF.MC', 'REP.MC', 'BBVA.MC', 'SCYR.MC',
            'MEL.MC', 'ACS.MC', 'IAG.MC'] 
_cont = 0
for _t in _tickers:
    _temp_df = data_retrieve.get_data_yfinance(_t)
    _temp_df.index = _temp_df.index.date.copy()
    _temp_df_adj = data_retrieve.dividends_norm_ret(_temp_df['Close'], _temp_df['Dividends'])
    _temp_df_adj.dropna(inplace = True)
    _temp_close = _temp_df_adj[['Close']].copy()
    _temp_div = _temp_df_adj[['Dividends']].copy()
    _temp_adj_ret = _temp_df_adj[['Adj_Ret']].copy()
    _temp_close.columns = [_t + '_Close']
    _temp_div.columns = [_t + '_Div']
    _temp_adj_ret.columns = [_t + '_Adj_Ret']
    
    if _cont == 0:
        _temp_close_final = _temp_close.copy()
        _temp_div_final = _temp_div.copy()
        _temp_adj_ret_final = _temp_adj_ret.copy()
        
    else:
        _temp_close_final =  _temp_close_final.merge(_temp_close, left_index = True,
                                                     right_index = True, how = 'left')
        _temp_div_final =  _temp_div_final.merge(_temp_div, left_index = True,
                                                 right_index = True, how = 'left')
        _temp_adj_ret_final =  _temp_adj_ret_final.merge(_temp_adj_ret, left_index = True,
                                                         right_index = True, how = 'left')
    _cont = _cont + 1

_temp_close_final.dropna(inplace = True)
_temp_div_final.dropna(inplace = True)
_temp_adj_ret_final.dropna(inplace = True)

master_instruments = {}

for _t in _tickers:
    _df = pd.DataFrame(columns = ['Close', 'Dividends', 'Adj_Returns'])
    _df['Close'] = _temp_close_final[_t + '_Close']
    _df['Dividends'] = _temp_div_final[_t + '_Div']
    _df['Adj_Returns'] = _temp_adj_ret_final[_t + '_Adj_Ret']
    
    master_instruments.update({_t: _df})



"""----------------------------------------------------------------------------
Model input construction
----------------------------------------------------------------------------"""
_balance = 10
master_ret = pd.DataFrame(columns = _tickers)
for _k in _tickers:
    master_ret[_k] = master_instruments[_k]['Adj_Returns']
master_ret = master_ret.rolling(_balance).sum().shift()
ind = np.arange(_balance, len(master_ret) - 1, _balance)
_fechas = master_ret.index[ind]
master_ret = master_ret.iloc[ind, :]
master_ret = master_ret

master_std = pd.DataFrame(columns = _tickers)
for _k in _tickers:
    master_std[_k] = master_instruments[_k]['Adj_Returns']
master_std = master_std.rolling(_balance).std().shift()
ind = np.arange(_balance, len(master_std) - 1, _balance)
_fechas = master_std.index[ind]
master_std = master_std.iloc[ind, :]
master_std = master_std

_master_corr_temp = pd.DataFrame(columns = _tickers)
for _k in _tickers:
    _master_corr_temp[_k] = master_instruments[_k]['Adj_Returns']
master_corr = np.zeros((master_std.shape[0], len(_tickers), len(_tickers)))
for i in range(0, master_corr.shape[0]):
    master_corr[i, :, :] = _master_corr_temp.iloc[i : (i + _balance), :].corr()


"""----------------------------------------------------------------------------
Model input construction
----------------------------------------------------------------------------"""

t = np.arange(0, len(_tickers)).tolist()
comb = list(itertools.combinations(t, 4))
unq = list(set(comb))


sh_total = {}
pl_total = {}
loss_total = {}
tickers_total = {}
w_or_total = {}
w_pred_total = {}

for contador_grande in range(0, len(unq)):
    print('Contador: ' + str(contador_grande))
    print("===============================")
    _t = list(unq[contador_grande])
    _tickers_fin = [_tickers[i] for i in _t]

    print(_tickers_fin)
    print("===============================")
    tickers_total.update({contador_grande: _tickers_fin})
    
    master_ret_temp = master_ret[_tickers_fin].copy()
    master_std_temp = master_std[_tickers_fin].copy()
    master_corr_temp = np.zeros((master_corr.shape[0], len(_tickers_fin), len(_tickers_fin)))
    for _nprof in range(0, master_corr.shape[0]):
        master_corr_temp[_nprof, :, :] = master_corr[_nprof, :, :][_t, :][:, _t].copy()
        
        
    _historic = 10
    master_corr_cube_x = np.zeros((master_corr_temp.shape[0] - _historic, _historic, master_corr_temp.shape[1], 
                                master_corr_temp.shape[2]))
    master_corr_cube_y = np.zeros((master_corr_temp.shape[0] - _historic, master_corr_temp.shape[1], 
                                master_corr_temp.shape[2]))
    
    for i in range(0, master_corr_cube_x.shape[0]):
        master_corr_cube_x[i, :, :, :] = master_corr_temp[i : (i + _historic), :, :]
        master_corr_cube_y[i, :, :] = master_corr_temp[i + _historic, :, :]
    
    _corr_x_train, _corr_x_val, _corr_x_test, _corr_y_train, _corr_y_val, _corr_y_test = model_inputs.train_test_val_split(
      master_corr_cube_x, master_corr_cube_y, 0.8, 0.1, 0.1, 2, False)
    fechas_predict_corr = _fechas[-(_corr_y_test.shape[0]):]
    print(_corr_x_test.shape)
    print("===============================")
    
    model = model_inputs.get_model_corr(_corr_x_train, _historic)
    model.summary()
    _epochs = 50
    _history_2 = model.fit(_corr_x_train, 
                          _corr_y_train,
                          epochs = _epochs,
                          validation_data = (_corr_x_val, _corr_y_val),
                          verbose = True)
    _corr_y_pred = model.predict(_corr_x_test)
    for k in range(0, _corr_y_pred.shape[0]):
      for j in range(0, _corr_y_pred.shape[1]):
        _temp = _corr_y_pred[k, :, :].copy()
        _corr_y_pred[k, :, :] = (_temp + np.transpose(_temp)) / 2
        _corr_y_pred[k, j, j] = 1
    fechas_predict_corr = _fechas[-(_corr_y_pred.shape[0]):]
    
    loss_t = pd.DataFrame(columns = ['Train','Val'])
    loss_t['Train'] = _history_2.history['loss']
    loss_t['Val'] = _history_2.history['val_loss']
    loss_total.update({contador_grande: loss_t})

    original_ret = master_ret_temp.loc[fechas_predict_corr].copy()
    original_std = master_std_temp.loc[fechas_predict_corr].copy()
    original_cov = np.zeros((_corr_y_test.shape[0], len(_tickers_fin), len(_tickers_fin)))
    for _tem in range(0, original_cov.shape[0]):
        _m_cov_t = _corr_y_test[_tem, :, :].copy()
        for _nrow in range(0, _m_cov_t.shape[0]):
            for _ncol in range(0, _m_cov_t.shape[1]):
                _m_cov_t[_nrow, _ncol] = _corr_y_test[_tem, _nrow, _ncol] * original_std.iloc[_tem, _nrow] * original_std.iloc[_tem, _ncol]
        original_cov[_tem, :, :] = _m_cov_t.copy()

    final_ret = master_ret_temp.loc[fechas_predict_corr].copy()
    final_std = master_std_temp.loc[fechas_predict_corr].copy()
    final_cov = np.zeros((_corr_y_pred.shape[0], len(_tickers_fin), len(_tickers_fin)))
    for _tem in range(0, final_cov.shape[0]):
        _m_cov_t = _corr_y_pred[_tem, :, :].copy()
        for _nrow in range(0, _m_cov_t.shape[0]):
            for _ncol in range(0, _m_cov_t.shape[1]):
                _m_cov_t[_nrow, _ncol] = _corr_y_pred[_tem, _nrow, _ncol] * final_std.iloc[_tem, _nrow] * final_std.iloc[_tem, _ncol]
        final_cov[_tem, :, :] = _m_cov_t.copy()

    
    w_or = np.zeros((len(fechas_predict_corr), len(_tickers_fin)))
    for i in range(0, original_cov.shape[0]):
        sol = sharpe_functions.max_sharpe_ratio(original_ret.iloc[i, ], original_cov[i, :, :], 0)
        a = sol.x
        a[a < 0.001] = 0
        w_or[i, :] = a 

    w_pred = np.zeros((len(fechas_predict_corr), len(_tickers_fin)))
    for i in range(0, final_cov.shape[0]):
        sol = sharpe_functions.max_sharpe_ratio(final_ret.iloc[i, ], final_cov[i, :, :], 0)
        a = sol.x
        a[a < 0.001] = 0
        w_pred[i, :] = a

    #Ya tengo los pesos
    w_or = pd.DataFrame(w_or, columns = _tickers_fin, index = fechas_predict_corr)
    w_pred = pd.DataFrame(w_pred, columns = _tickers_fin, index = fechas_predict_corr)
    
      
    precios_ajustados = pd.DataFrame(columns = _tickers_fin)
    for _c in precios_ajustados.columns:
        _temp = master_instruments[_c].copy()
        _temp['Adj_Close'] = _temp['Close']
        for _conti2 in range(1, len(_temp)):
            _temp['Adj_Close'].iloc[_conti2] = _temp['Adj_Close'].iloc[(_conti2 - 1)] * np.exp(_temp['Adj_Returns'].iloc[_conti2])
        precios_ajustados[_c] = _temp['Adj_Close']
      
    aa = pd.DataFrame(columns = ['Predicted', 'Original'])
    aa['Predicted'] = sharpe_functions.pl(w_pred, precios_ajustados)
    aa['Original'] = sharpe_functions.pl(w_or, precios_ajustados)
    
    sh = pd.DataFrame(columns = ['Predicted', 'Original'])
    sh['Predicted'] = sharpe_functions.sharpe_in_portfolio(w_or, original_ret, original_cov, np.arange(0, len(_tickers_fin)).tolist())
    sh['Original'] = sharpe_functions.sharpe_in_portfolio(w_pred, final_ret, final_cov, np.arange(0, len(_tickers_fin)).tolist())

    pl_total.update({contador_grande: aa})
    sh_total.update({contador_grande: sh})
    w_or_total.update({contador_grande: w_or})
    w_pred_total.update({contador_grande: w_pred})


    #Resultados en pantalla
    sh_results = pd.DataFrame(columns = ['Predicted', 'Original'], index = np.arange(0, len(sh_total)))
    print("===============================")
    for _cc in range(0, len(pl_total)):
      sh_results['Predicted'].iloc[_cc] = sh_total[_cc]['Predicted'].mean()
      sh_results['Original'].iloc[_cc] = sh_total[_cc]['Original'].mean()
    print(sh_results)
    print("===============================")
    print("Loss validation: " + str(loss_t['Val'].iloc[-1]))
    print("===============================")
    print((sh_results['Predicted'] > sh_results['Original']).sum() / len(sh_total))
    print("===============================")
    print(sh_results.mean())
    print("===============================")
    print("###############################")

    #Guardamos en Drive dentro del bucle así no se pierde
    with open('/content/drive/MyDrive/sh_total_loop3.pickle', 'wb') as handle:
      pickle.dump(sh_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/content/drive/MyDrive/pl_total_loop3.pickle', 'wb') as handle:
        pickle.dump(pl_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/content/drive/MyDrive/loss_total_loop3.pickle', 'wb') as handle:
        pickle.dump(loss_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/content/drive/MyDrive/tickers_total_loop3.pickle', 'wb') as handle:
        pickle.dump(tickers_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/content/drive/MyDrive/w_or_total_loop3.pickle', 'wb') as handle:
        pickle.dump(w_or_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/content/drive/MyDrive/w_pred_total_loop3.pickle', 'wb') as handle:
        pickle.dump(w_pred_total, handle, protocol=pickle.HIGHEST_PROTOCOL)  





