import pandas as pd
import numpy as np
import scipy.optimize as sco

def sharpe_in_portfolio(_weights, _original_ret, _original_cov, _ind_t):
  
  df_sharpe = pd.DataFrame(columns = ['Sharpe'], index = _weights.index)
  
  for i in range(0, len(_weights) - 1):
      
      _ind = list(_ind_t)
      _ind = [int(x) for x in _ind]
      
      ticks = _weights.columns
      _ret = _original_ret[ticks].iloc[(i + 1)]
      _cov = _original_cov[:, _ind, :][:, :, _ind][(i + 1), :, :]
      
      df_sharpe.loc[_weights.index[i]] = - neg_sharpe_ratio(_weights.iloc[i, :], _ret, _cov, 0)
  df_sharpe = df_sharpe.iloc[:-1].copy()
  
  return df_sharpe

def portfolio_annualised_performance(_w, _mu, _cov):
    _mu_p = np.sum(_mu * _w)
    _std_p = np.sqrt(np.dot(_w.T, np.dot(_cov, _w)))
    return _std_p, _mu_p

def neg_sharpe_ratio(_w, _mu, _cov, _r):
    _std_p, _mu_p = portfolio_annualised_performance(_w, _mu, _cov)
    return -(_mu_p - _r) / _std_p


# ==============================================================================

def max_sharpe_ratio(_mu, _cov, _r):
    num_assets = len(_mu)
    args = (_mu, _cov, _r)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def pl(pesos, p_ajustados):
  p_ajustados = p_ajustados.loc[pesos.index]
  nominal_t = 10000
  pl_fin = []
  pl_fin.append(nominal_t)
  for _conti in range(0,len(pesos) - 1):
      nominal = nominal_t
      nominal_t = (nominal * pesos.iloc[_conti, :] / p_ajustados.iloc[_conti, :] * p_ajustados.iloc[_conti + 1, :]).sum()
      pl_fin.append(nominal_t)
  return pl_fin