import pandas as pd
import numpy as np
import statsmodels.api as sm
from zads import FactorUtil, Frequency, Instrument, level2
from scipy.stats import kurtosis, skew
import pprint

# NOTICE: 此处说明因子引用的研报以及阐述其逻辑
# alpha_MEMO
# 来源 :  20240609-开源证券-市场微观结构研究系列（25）：订单流系列，挂单方向长期记忆性的讨论与应用
# 
# 逻辑 : 
#22年后A股的订单的长时记忆性尤其明显，而笔者认为这源自于机构的算法拆单行为。因此，计算每日逐笔数据的
#自相关系数，并且相对于滞后阶数的对数值做回归，从而得到的截距值作为长时记忆强度因子_LMS，用于衡量该
#股票每日的长时记忆强度。并且，将20日平滑作为选股指标。
#选股逻辑：
#长时记忆强度反应了机构特征的明显与否，一般认为机构在选股的质量上整体优于散户，从而形成了多头和空头分组
#的相对差异，可以归因于机构交易者相对于散户的信息优势。
#高维记忆_MEMO因子来自于对LMS的改进，由线性模式转为统计模型，同时筛选订单样本和补充价格和数量的信息。
#首先，将时序样本缩短至最后半小时，然后计算订单流挂单方向滞后 1 至 100 阶的自相关系数作为统计分布，最
#后分别计算该分布的峰度和偏度指标，在截面上等权合成为最终信号。


def MEMO_by_inst_date(inst_code:str, date:str):
    # 获取level2数据，如果空返回nan
    inst_date_trans_df = level2(inst_code, date)
    if (inst_date_trans_df is None) or (inst_date_trans_df.empty):
        return date, np.NaN, np.NaN
    
    # 日成交量，停牌返回nan
    volume = inst_date_trans_df['Volume'].sum()
    if volume==0: # 停牌
        return date, volume, np.NaN

    #获取交易日最后半个小时的逐笔数据
    # 将 'Time' 列转换为 datetime 对象
    inst_date_trans_df['Time'] = pd.to_datetime(inst_date_trans_df['Time'], format='%H:%M:%S')

    # 筛选出交易日最后半个小时的数据
    # 注意：这里假设您的数据已经是按日期排序的
    inst_date_trans_df = inst_date_trans_df[inst_date_trans_df['Time'] >= pd.to_datetime('1900-01-01 14:30:00')]
    
    if len(inst_date_trans_df) == 0:
        return date, volume, np.NaN
    

    #创建订单方向序列，买入标记为1，卖出标记为-1
    inst_date_trans_df.loc[inst_date_trans_df['Type'] == 'B', 'Type'] = 1
    inst_date_trans_df.loc[inst_date_trans_df['Type'] == 'S', 'Type'] = -1
    #订单方向序列
    dir_date_df = pd.DataFrame()
    # 将 'Type' 列转换为数值类型，无法转换的值将被设置为 NaN
    dir_date_df['Type'] = pd.to_numeric(inst_date_trans_df['Type'], errors='coerce')

    # 检查数据中的无效值
    if dir_date_df['Type'].isnull().any():
        dir_date_df['Type'] = dir_date_df['Type'].dropna()  # 删除含有NaN的行

    #创建滞后值作为自变量
    if len(dir_date_df) >= 100:
        lags = 100
    else:
        lags = len(dir_date_df)


    def calculate(lags):
        # 假设 dir_date_df 是包含时间序列数据的 DataFrame，lags 是滞后阶数
    # 计算自相关系数之前，检查数据长度
        if len(dir_date_df['Type']) < lags:
            print(f"警告：数据长度 {len(dir_date_df['Type'])} 小于滞后阶数 {lags}。")
            
        # 根据数据长度调整滞后阶数或获取更多数据
        #lags = len(dir_date_df['Type']) - 1 

        #计算自相关系数，输出数组（索引0相当于滞后1）
        acf_values = sm.tsa.acf(dir_date_df['Type'],nlags=lags)
        # 将自相关系数转换为 Series，并用滞后阶数作为索引
        try:
            if len(acf_values) == lags +1:
                acf_series = pd.Series(acf_values[1:], index=np.arange(1, lags + 1))
            else:
                acf_series = pd.Series(acf_values, index=np.arange(1, lags + 1))
        except:
            return np.NaN

        #计算偏度和峰度
        #计算偏度
        skewness = skew(acf_series)
        #计算常规峰度
        kurt = kurtosis(acf_series)
        kurtosis_regular = kurt + 3
        return -(skewness + kurtosis_regular)
    
    MEMO = calculate(lags)

    return date, volume, MEMO



# NOTICE: 此处实现单合约的时序因子逻辑
# 1. 第1个参数必需定义：inst_obj:Instrument
def MEMO_by_inst(inst_obj:Instrument, rolling_window:int=20):
    # 交易日期和合约代码
    date_list = inst_obj.kline_day['time'].astype(str).unique()
    inst_code = inst_obj.code
    # 计算 卖单集中度（未平均20天）
    MEMO_list = [MEMO_by_inst_date(inst_code, date) for date in date_list]
    daily_df = pd.DataFrame(MEMO_list).rename(columns={0:'time',1:'volume',2:'MEMO'})
    daily_df.sort_values(by='time', inplace=True)
    daily_df['inst'] = inst_code

    daily_df['MEMO'] = daily_df['MEMO']\
        .rolling(rolling_window, min_periods=int(rolling_window*0.5)).mean()

    return daily_df[['inst','time','volume','MEMO']]

# NOTICE: 对合约的时序因子进行横截面处理
# 1. 前3个参数必需定义，且顺序不可变更: factor_df, freq
# 2. 实现因子逻辑时默认输出日频因子
# 3. 做横截面处理，需注意处理方法的顺序
def MEMO(
    factor_df:pd.DataFrame,
    freq:Frequency=Frequency.DAILY):

    # 对数调整
    # 将停牌时的因子值设为NA
    factor_df['MEMO'] = np.where(factor_df['volume']<=0, np.NaN, factor_df['MEMO'])
    keep_columns = ['inst','time','MEMO']
    factor_df.drop(columns=set(factor_df.columns)-set(keep_columns), inplace=True)

    # !!! NOTICE: 因子重采样
    #     适用场景: 高频转低频
    #     譬如 factor_df 为因子, 但是最终想输出月频因子
    if(freq>Frequency.DAILY):
        factor_df = FactorUtil.resample(factor_df=factor_df, freq=freq)

    factor_df.rename(columns={'MEMO':'factor_value'}, inplace=True)
    return factor_df[['inst','time','factor_value']]
