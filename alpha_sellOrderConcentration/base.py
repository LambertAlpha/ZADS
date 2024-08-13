import pandas as pd
import numpy as np
from zads import FactorUtil, Frequency, Instrument, level2

# NOTICE: 此处说明因子引用的研报以及阐述其逻辑
# alpha_sellOrderConcentration :
# 来源 :  20191107-海通证券-选股因子系列研究（五十六）-- 买卖单数据中的Alpha-c 卖单集中度
# 
# 逻辑 : 
# 在逐笔数据中，投资者往往比较关注 BS 标志，并常常围绕该字段构建因子。
# 然而，除了 BS 标志外，叫卖序号以及叫买序号同样值得关注。
# 我们可基于逐笔数据中的叫卖序号以及叫买序号合成得到买卖单数据，
# 并基于买卖单数据构建选股因子。
# 
# 基于各股票的买卖单数据，计算：
# 卖单集中度 = 过去T=20日均值(sum(卖单成交金额^2) / 总成交金额^2)
# 计算因子值时进行对数调整

def sellOrderConcentration_by_inst_date(inst_code:str, date:str):
    # 获取level2数据，如果空返回nan
    inst_date_trans_df = level2(inst_code, date)
    if (inst_date_trans_df is None) or (inst_date_trans_df.empty):
        return date, np.NaN, np.NaN
    
    # 日成交量，停牌返回nan
    volume = inst_date_trans_df['Volume'].sum()
    if volume==0: # 停牌
        return date, volume, np.NaN

    # 成交金额
    inst_date_trans_df['Amt'] = inst_date_trans_df['Volume'].values * inst_date_trans_df['Price'].values
    # 总成交金额
    total_amt = inst_date_trans_df['Amt'].sum()

    # 卖单数据
    sell_order_df = inst_date_trans_df[['SaleOrderID','Amt']].groupby('SaleOrderID')\
        .agg({'Amt':'sum'}).reset_index()

    # 卖单集中度（未平均20天）
    sellOrderConcentration = (sell_order_df['Amt'].values ** 2).sum() / (total_amt ** 2)

    return date, volume, sellOrderConcentration


# NOTICE: 此处实现单合约的时序因子逻辑
# 1. 第1个参数必需定义：inst_obj:Instrument
def sellOrderConcentration_by_inst(inst_obj:Instrument, rolling_window:int=20):
    # 交易日期和合约代码
    date_list = inst_obj.kline_day['time'].astype(str).unique()
    inst_code = inst_obj.code
    # 计算 卖单集中度（未平均20天）
    sellOrderConcentration_list = [sellOrderConcentration_by_inst_date(inst_code, date) for date in date_list]
    daily_df = pd.DataFrame(sellOrderConcentration_list).rename(columns={0:'time',1:'volume',2:'sellOrderConcentration'})
    daily_df.sort_values(by='time', inplace=True)
    daily_df['inst'] = inst_code

    daily_df['sellOrderConcentration'] = daily_df['sellOrderConcentration']\
        .rolling(rolling_window, min_periods=int(rolling_window*0.5)).mean()

    return daily_df[['inst','time','volume','sellOrderConcentration']]


# NOTICE: 对合约的时序因子进行横截面处理
# 1. 前3个参数必需定义，且顺序不可变更: factor_df, freq
# 2. 实现因子逻辑时默认输出日频因子
# 3. 做横截面处理，需注意处理方法的顺序
def sellOrderConcentration(
    factor_df:pd.DataFrame,
    freq:Frequency=Frequency.DAILY):

    # 对数调整
    # 将停牌时的因子值设为NA
    factor_df['sellOrderConcentration'] = np.where((factor_df['volume']<=0), np.NaN, np.log(factor_df['sellOrderConcentration']))
    keep_columns = ['inst','time','sellOrderConcentration']
    factor_df.drop(columns=set(factor_df.columns)-set(keep_columns), inplace=True)

    # !!! NOTICE: 因子重采样
    #     适用场景: 高频转低频
    #     譬如 factor_df 为因子, 但是最终想输出月频因子
    if(freq>Frequency.DAILY):
        factor_df = FactorUtil.resample(factor_df=factor_df, freq=freq)

    factor_df.rename(columns={'sellOrderConcentration':'factor_value'}, inplace=True)
    return factor_df[['inst','time','factor_value']]
