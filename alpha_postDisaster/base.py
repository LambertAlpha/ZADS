import pandas as pd
import numpy as np
from zads import CommonUtil, Instrument, Frequency, FactorUtil

# NOTICE: 此处说明因子引用的研报以及阐述其逻辑
# alpha_postDisaster :
# 来源 : 20220530-方正证券-多因子选股系列研究之三 –– "灾后重建"因子
# 
# 逻辑 : 
# 由于收益波动比降低而带来的抛售，对于股票的股价来说是一种“灾
# 难”，而那些在“灾难”后买入建仓，参与重建的人，将享受到后续
# 补涨带来的丰厚回报
# 
# （1）剔除开盘与收盘部分的信息，计算"更优波动率": 第 t-4、t-3、
# t-2、t-1、t 分钟的分钟开盘价、分钟最高价、分钟最低价和分钟收盘
# 价，共 20 个价格数据求标准差，然后除以这20 个价格数据的均值，
# 将该比值取平方，作为 t 分钟的“更优波动率”
# （2）计算每分钟的收益波动比：用 t 分钟的收益率与 t 分钟的“更优
# 波动率”的比值，作为 t 分钟的收益波动比
# （3）求每天的收益波动比序列与“更优波动率”序列之间的协方差，求最
# 近 20 天的均值和标准差，得到“月均重建”因子和“月稳重建”因子，并
# 将二者等权合成为“灾后重建”因子


def rolling_get_better_volt(
    rolling_df_min:pd.DataFrame, 
    standard_length:int=5):

    if len(rolling_df_min) < standard_length:
        return np.NaN

    whole_20_mean = rolling_df_min[['high','low','open','close']].values.mean()
    whole_20_std = rolling_df_min[['high','low','open','close']].values.std()
    better_volt = np.power(whole_20_std/whole_20_mean, 2)

    return better_volt


def postDisaster_by_inst_date(inst_df_min_date:pd.DataFrame, date:str):

    volume = inst_df_min_date['volume'].sum()
    if volume==0: # 停牌
        return date, volume, np.NaN
    
    inst_df_min_date.sort_index(inplace=True)
    
    inst_df_min_date['return'] = inst_df_min_date['close'].values / inst_df_min_date['close'].shift(1).values -1

    # 剔除开盘和收盘数据
    inst_df_min_date.drop(index=[inst_df_min_date.index[0],inst_df_min_date.index[-1]], inplace=True)

    # 计算“更优波动率”
    window = 5
    inst_df_min_date['better_volt'] = CommonUtil.roll_reduce(inst_df_min_date,
        window=window,
        func=lambda df,standard_length= window: rolling_get_better_volt(df,standard_length))

    # 用 t 分钟的收益率与 t 分钟的“更优波动率”的比值，作为 t 分钟的收益波动比
    def calc_ret_volt_ratio(s_return, s_better_volt):
        if(s_better_volt==0):
            return np.NaN
        if(s_return==0):
            return 0.0
        return s_return/s_better_volt
    inst_df_min_date['ret_volt_ratio'] = np.vectorize(calc_ret_volt_ratio)(inst_df_min_date['return'],inst_df_min_date['better_volt'])

    # 求每天的收益波动比序列与“更优波动率”序列之间的协方差
    tgt_cov = inst_df_min_date['ret_volt_ratio'].cov(other=inst_df_min_date['better_volt'])
    return date, volume, tgt_cov


def rolling_mean_process(x_sr:pd.Series, standard_length:int=20):

    if len(x_sr) < standard_length:
        return np.NaN
    if len(x_sr[x_sr.isna()])/len(x_sr)>0.6:
        return np.NaN
    else:
        x_sr = x_sr[~x_sr.isna()]
        return x_sr.mean()


def rolling_std_process(x_sr:pd.Series, standard_length:int=20):

    if len(x_sr) < standard_length:
        return np.NaN
    if len(x_sr[x_sr.isna()])/len(x_sr)>0.6:
        return np.NaN
    else:
        x_sr = x_sr[~x_sr.isna()]
        return x_sr.std()


# NOTICE: 此处实现单合约的时序因子逻辑
# 1. 第1个参数必需定义：inst_obj:Instrument
# 2. 其他参数可自由按需定义, 但必须给定默认值
# 3. 返回值将会被用于 reduce (对合约的时序因子进行横截面处理) 
def postDisaster_by_inst(inst_obj:Instrument, rolling_window:int=20):
    # NOTICE: 若需对每天的分钟频行情进行处理
    # 可使用 inst_obj.kline_min_date_by_date() 生成 iterator
    # 该 iterator 在每次迭代中返回 (date, df) ==> (日期，分钟频行情)
    daily_df_list = [postDisaster_by_inst_date(df, date) for date, df in inst_obj.kline_min_date_by_date()]
    daily_df = pd.DataFrame(daily_df_list).rename(columns={0:'time',1:'volume',2:'tgt_cov'})
    daily_df.sort_values(by='time', inplace=True)
    daily_df['inst'] = inst_obj.code

    # 求最近 20 天的均值和标准差，得到“月均重建”因子和“月稳重建”因子
    daily_df['monthly_mean'] = daily_df['tgt_cov'].rolling(rolling_window,min_periods=1)\
        .apply(lambda x :rolling_mean_process(x,rolling_window))
    daily_df['monthly_std'] = daily_df['tgt_cov'].rolling(rolling_window,min_periods=1)\
        .apply(lambda x :rolling_std_process(x,rolling_window))

    return daily_df[['inst','time','volume','monthly_mean','monthly_std']]


# NOTICE: 对合约的时序因子进行横截面处理
# 1. 前2个参数必需定义, 且顺序不可变更: factor_df, freq
# 2. 其他参数可自由按需定义, 但必须给定默认值
# 3. 做横截面处理, 需注意处理方法的顺序, 严格按照因子逻辑进行处理
def postDisaster(
    factor_df:pd.DataFrame,
    freq:Frequency=Frequency.DAILY):

    # 横截面z-score标准化
    factor_df = FactorUtil.crssct_stddzt(
        factor_df, 
        ['monthly_mean','monthly_std']) 

    # 将二者等权合成为“灾后重建”因子
    #     将停牌时的因子值设为NA
    factor_df['PostDisaster'] = np.where(
        (factor_df['volume']<=0),
        np.NaN,
        factor_df['monthly_mean'] + factor_df['monthly_std'])

    # !!! NOTICE: 因子重采样
    #     适用场景: 高频转低频
    #     譬如 factor_df 为日频因子, 但是最终想输出月频因子
    if(freq>Frequency.DAILY):
        factor_df = FactorUtil.resample(factor_df=factor_df, freq=freq)

    # 横截面 MAD 去极值
    factor_df = FactorUtil.crssct_MAD(factor_df, ['PostDisaster'])

    factor_df.rename(columns={'PostDisaster':'factor_value'}, inplace=True)
    return factor_df[['inst','time','factor_value']]
