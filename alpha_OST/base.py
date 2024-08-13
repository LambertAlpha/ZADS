import pandas as pd
import numpy as np
import statsmodels.api as sm
from zads import FactorUtil, Frequency, Instrument, kline_day, level2
from scipy.stats import kurtosis
from scipy.signal import find_peaks

# NOTICE: 此处说明因子引用的研报以及阐述其逻辑
# alpha_OST
# 来源 :  20240609-开源证券-市场微观结构研究系列（25）：订单流系列，挂单方向长期记忆性的讨论与应用
# 
# 逻辑 : 
#22年后A股的订单的长时记忆性尤其明显，而笔者认为这源自于机构的算法拆单行为。因此，计算每日逐笔数据的
#自相关系数，并且相对于滞后阶数的对数值做回归，从而得到的截距值作为长时记忆强度因子_LMS，用于衡量该
#股票每日的长时记忆强度。并且，将20日平滑作为选股指标。
#选股逻辑：
#长时记忆强度反应了机构特征的明显与否，一般认为机构在选股的质量上整体优于散户，从而形成了多头和空头分组
#的相对差异，可以归因于机构交易者相对于散户的信息优势。
#优化：利用傅里叶变换将挂单方向的时域信息转变为更容易刻画周期性的频域
#特征，然后再统计频域强度的峰度作为日频信号并平滑 20 日作为最终因子。与前文
#的处理有所不同，我们通过分析不同类型委托订单的特点，发现小额委托的长期记
#强度在区分股票间“拆单”行为强度上更准确。因而，在因子的计算步骤中，笔者
#增加了一步样本筛选的处理，只选取当日委托数量较小的 50%订单用于转换频域信
#息，并将上述计算得到的因子命名为分拆痕迹_OST 因子。

def OST_by_inst_date(inst_code:str, date:str):
    # 获取level2数据，如果空返回nan
    inst_date_trans_df = level2(inst_code, date)
    if (inst_date_trans_df is None) or (inst_date_trans_df.empty):
        return date, np.NaN, np.NaN
    
    # 日成交量，停牌返回nan
    volume = inst_date_trans_df['Volume'].sum()
    if volume==0: # 停牌
        return date, volume, np.NaN

    #选取当日委托数量较小的 50%订单
    sorted_df = inst_date_trans_df.sort_values(by='Volume', ascending=True)
    sorted_df = sorted_df.head(len(sorted_df)//2)

    # 现在 "Time" 列是索引，根据索引来排序 DataFrame
    # 假设您要按照时间顺序进行排序
    inst_date_trans_df = sorted_df.sort_values(by="Time", ascending=True) #升序

    #创建订单方向序列，买入标记为1，卖出标记为-1
    print(inst_date_trans_df['Type'])
    inst_date_trans_df['Type'] = inst_date_trans_df['Type'].replace({'B':1, 'S':-1})
    print(inst_date_trans_df['Type'])

    print('------------------------------------------------------')
    #订单方向序列
    dir_date_df = pd.DataFrame()
    # 将 'Type' 列转换为数值类型，无法转换的值将被设置为 NaN
    dir_date_df['Type'] = pd.to_numeric(inst_date_trans_df['Type'], errors='coerce')
    dir_date_df['Time'] = inst_date_trans_df['Time']
    dir_date_df = dir_date_df.set_index('Time')
    print(dir_date_df)

    #傅立叶变换
    fft_result = np.fft.fft(dir_date_df)
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

    #计算振幅
    amplitudes = np.abs(fft_result)
    print(amplitudes)

    #计算峰度
    # 由于FFT结果是对称的，我们只取一半用于计算
    amplitudes_half = amplitudes[:len(amplitudes)//2]
    kurtosis_value = kurtosis(amplitudes_half)

    OST = kurtosis_value
    print(OST)
    return date, volume, OST


# NOTICE: 此处实现单合约的时序因子逻辑
# 1. 第1个参数必需定义：inst_obj:Instrument
def OST_by_inst(inst_obj:Instrument, rolling_window:int=20):
    # 交易日期和合约代码
    date_list = inst_obj.kline_day['time'].astype(str).unique()
    inst_code = inst_obj.code
    # 计算 卖单集中度（未平均20天）
    OST_list = [OST_by_inst_date(inst_code, date) for date in date_list]
    daily_df = pd.DataFrame(OST_list).rename(columns={0:'time',1:'volume',2:'OST'})
    daily_df.sort_values(by='time', inplace=True)
    daily_df['inst'] = inst_code

    daily_df['OST'] = daily_df['OST']\
        .rolling(rolling_window, min_periods=int(rolling_window*0.5)).mean()

    return daily_df[['inst','time','volume','OST']]

# NOTICE: 对合约的时序因子进行横截面处理
# 1. 前3个参数必需定义，且顺序不可变更: factor_df, freq
# 2. 实现因子逻辑时默认输出日频因子
# 3. 做横截面处理，需注意处理方法的顺序
def OST(
    factor_df:pd.DataFrame,
    freq:Frequency=Frequency.DAILY):

    # 对数调整
    # 将停牌时的因子值设为NA
    factor_df['OST'] = np.where(factor_df['volume']<=0, np.NaN, factor_df['OST'])
    keep_columns = ['inst','time','OST']
    factor_df.drop(columns=set(factor_df.columns)-set(keep_columns), inplace=True)

    # !!! NOTICE: 因子重采样
    #     适用场景: 高频转低频
    #     譬如 factor_df 为因子, 但是最终想输出月频因子
    if(freq>Frequency.DAILY):
        factor_df = FactorUtil.resample(factor_df=factor_df, freq=freq)

    factor_df.rename(columns={'OST':'factor_value'}, inplace=True)
    return factor_df[['inst','time','factor_value']]
