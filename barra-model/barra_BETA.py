import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool as ProcessPool
from functools import partial
import statsmodels.api as sm
import pandas as pd
import numpy as np
from zads import FactorUtil, CommonUtil, Frequency, Instrument
from tqdm import tqdm
from typing import List
import gc

# 股票超额收益日序列和市值加权指数超额收益日序列进行WLS的回归系数，
# beta表示股票相对于指数涨跌的弹性大小。
# 其中rf是无风险收益日序列，rt是股票收益日序列，Rt是市值加权指数
# （此处用中证500）超额收益日序列，回归系数采取过去252交易日的收益
# 数据，回归权重采用指数加权移动平均，半衰期为63个交易日



def rolling_get_BETA(
    rolling_df:pd.DataFrame, 
    standard_length:int=252, 
    time_weights:np.array=None):

    if len(rolling_df) < standard_length:
        return np.NaN

    if len(rolling_df[rolling_df['volume']==0])/len(rolling_df) > 0.5:
        return np.NaN

    # # 计算时序指数加权权重
    # omega = 0.5**(1/decay)
    # weights = np.array([omega**(standard_length-i) for i in range(1,standard_length+1)])
    rolling_df['time_weight'] = time_weights
    
    rolling_df = rolling_df[rolling_df['volume']!=0]

    y_col = 'excess_daily_return'
    x_col = 'market_excess_return'

    # 将当前y列或x列为空的行和不为空的行分开处理
    df_reg = rolling_df[(~rolling_df['excess_daily_return'].isna()) & (~rolling_df['market_excess_return'].isna())].copy(deep=True)

    x = sm.add_constant(df_reg[x_col])
    y = df_reg[y_col]
    w = df_reg['time_weight']

    if len(df_reg)==0:
        beta = np.NaN
    else:
        result = sm.WLS(y, x, weights=w).fit()
        beta = result.params[0]

    return beta



def BETA_by_inst(
    inst_df:pd.DataFrame,
    rolling_window:int=252,
    weights:np.array=None)-> pd.DataFrame :

    if len(inst_df)< rolling_window:
        inst_df.reset_index(drop=True, inplace=True)
        inst_df['factor_value'] = np.nan
        return inst_df[['inst','time','volume','factor_value']]

    # inst_df.sort_index(inplace=True)
    inst_df.reset_index(drop=True, inplace=True)
    inst_df.sort_values(by='time',inplace=True)
    inst_df['excess_daily_return'] = inst_df['close'].values / inst_df['close'].shift(1).values -1 - 0.015

    #滚动计算因子值
    inst_df['factor_value'] = CommonUtil.roll_reduce(
        inst_df[['volume','excess_daily_return','market_excess_return']],
        window=rolling_window,
        func=lambda df,standard_length=rolling_window,time_weights=weights: 
            rolling_get_BETA(df,standard_length,time_weights))

    return inst_df[['inst','time','volume','factor_value']]



def BETA(
    whole_df:pd.DataFrame, 
    pool_symbol_df:pd.DataFrame,
    market_df:pd.DataFrame=None,
    rolling_window:int=252,
    decay:int=63,
    freq:Frequency=Frequency.DAILY)-> pd.DataFrame:

    market_df = market_df.sort_values(by=['inst','time'])
    market_df['market_excess_return'] = market_df['close'].values / market_df['close'].shift(1).values -1 -0.015

    whole_df = pd.merge(
        whole_df, 
        market_df[['time','market_excess_return']], 
        on=['time'], how='left')

    # 计算时序回归的指数加权权重
    omega = 0.5**(1/decay)
    weights = np.array([omega**(rolling_window-i) for i in range(1,rolling_window+1)])
    time_weights = weights/weights.sum()

    job_n=int(multiprocessing.cpu_count() // 2)
    partial_func = partial(
        BETA_by_inst, 
        rolling_window = rolling_window,
        weights = time_weights)
    res_df_list = []
    if(job_n > 1):
        with ProcessPool(processes=job_n) as pool:
            res_df_list = pool.map(
                partial_func, 
                [df for _,df in tqdm(factor_df.groupby('inst'), desc='beta_by_inst')])
    else:
        for _,df in tqdm(factor_df.groupby('inst'), desc='beta_by_inst'):
            res_df_list.append(partial_func(df))
    factor_df=pd.concat(res_df_list)

    # 过滤ST和不在合约池的因子
    start_day = factor_df['time'].min()
    end_day = factor_df['time'].max()
    pool_df = pool_symbol_df[(pool_symbol_df.index>=start_day)&(pool_symbol_df.index<=end_day)]
    factor_df = FactorUtil.filter_pool_and_st(
        factor_df=factor_df,
        pool_df=pool_df)

    ## 将停牌时的因子值设为NA
    factor_df['factor_value'] = np.where((factor_df['volume']<=0),np.NaN,factor_df['factor_value'])
    keep_columns = ['inst','time','factor_value']
    factor_df.drop(columns=set(factor_df.columns)-set(keep_columns), inplace=True)

    # 因子重采样
    factor_df = FactorUtil.resample(
        factor_df=factor_df,
        freq=freq)

    factor_df.sort_values(by=['time','inst'], inplace=True)

    return factor_df[['inst','time','factor_value']]
