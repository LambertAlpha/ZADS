import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool as ProcessPool
from functools import partial
import statsmodels.api as sm
import pandas as pd
import numpy as np
from zads import FactorUtil, CommonUtil, Frequency, Instrument, kline_day
from tqdm import tqdm
from typing import List
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm


'''
因子归因完整逻辑：(按照barra模型逻辑编写)
1. barra因子列表:
    1.) Liquidity: 已覆盖
    2.) Leverage: 已覆盖
    3.) Earning Variablity: VSAL(已完成), VERN(未完成), VFLO(未完成), SPIBS(未完成)
    4.) Earning Quality: 未覆盖
    5.) Profitability: 未覆盖
    6.) Investment Quality: 未覆盖
    7.) BTOP: 已覆盖
    8.) Earning Yield: 已覆盖
    9.) Long Term reversal: 未覆盖
    10.) Growth: 已覆盖
    11.) Momentum: 已覆盖
    12.) Mid Cap: 已覆盖
    13.) LSIZE: 已覆盖,使用NLSIZE
    14.) Beta: 已覆盖
    15.) Resival Volatility: 已覆盖
    16.) Dividend Yield: 已覆盖

2. 因子归因程序逻辑：
逻辑简述: 1. 先计算出每个因子每日的收益 2. 再将目标时间段内目标资产组合的每日收益回归到barra的20个因子上(OLS) 3. 回归所得的每个因子的系数即为该因子对资产组合收益的贡献

2.1.) 因子收益计算(barra原文算法):
因子每日的收益为全市场股票每日的收益回归到每个因子因子值后所得到每个因子的系数,此处使用的是WLS(加权最小二乘),权重为每支股票流通市值的平方根。
2.2.) 目标资产组合收益回归到barra因子上:
目标资产为根据某因子或某些因子构建出的多空资产组合中的第一组或第十组的收益序列，使用第一组还是第十组取决多空头，我们使用多头组。
2.3.) 回归得到因子系数：
最后,将投资组合的收益序列,依据OLS回归到barra的所有因子上,所得到的系数即为因子的贡献,同时我们得到p-value,根据p-value从小到大(统计显著性从大到小)排序画出因子贡献图，并保存。

3. 程序使用说明：
3.1.) 需要先运行barra所有因子目标时间段的因子值run_factor
3.2.) 需要运行目标因子(被分解为贡献的因子)目标时间段的backtest
3.3.) 在第73-76行替换为barra因子值目标时间段的文件路径(p1,p2,p3是老版run_factor,p4是新版run_factor)
3.4.) 在第124行替换为gen_return文件夹下run_factor.py在目标时间段跑出来的结果文件的路径
3.5.) 在第172行替换为目标因子backtest下第一组或第十组的assest文件路径


'''


def run_main():
# step1: 导入所有时序因子数据
    factor_lst = ['BLEV',
                'BTOP','CETOP','EGRO','RSTR','SGRO','STOM','HSIGMA','DASTD','LNCAP','BETA','DTOA',
                'ETOP','MLEV','NLSIZE', 'HALPHA',
                'ln_circ_mktcap','momentum_1m2y'
                # ,'VSAL', 暂时剔除了'CMRA' ‘STOQ’ ‘STOA’
                ]

    start_date = '20220331'
    end_date = '20240306'
    total_df = pd.DataFrame()
    for count, factor_name in enumerate(factor_lst):
        potential_path = []
        p1 = '/home/linboyi/barrra-model/risk_factors-master/barra_'+ factor_name +'/data/result/factor-' + factor_name + '-daily-stock-daily-post-' + start_date + '_' + end_date + '.csv'
        p2 = '/home/linboyi/barrra-model/risk_factors-master/barra_'+factor_name+'/data/result/factor_'+factor_name+'_stock_daily_daily_post/factor-'+factor_name+'-daily-stock-daily-post-'+ start_date + '_' + end_date + '.csv'
        p3 = '/home/linboyi/barrra-model/risk_factors-master/'+factor_name+'/data/result/factor-'+factor_name+'-daily-stock-daily-post-'+ start_date + '_' + end_date + '.csv'
        p4 = '/home/linboyi/barrra-model/risk_factors-master/barra_'+factor_name+'/data/result/factor_'+factor_name+'_daily_none/cross-stock.parquet.gz'
        potential_path.append(p1)
        potential_path.append(p2)
        potential_path.append(p3)
        potential_path.append(p4)

        for path in potential_path:
            if path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                    keep_columns = ['inst','time','factor_value']
                    df.drop(columns=set(df.columns)-set(keep_columns), inplace=True)
                    df.rename(columns={'factor_value': factor_name + '_factor_value'},inplace=True)

                    if count == 0:
                        total_df = df
                    else:
                        total_df = pd.merge(total_df,df,on=['inst','time'],how='left')
                    break
                except:
                    continue
            else:
                df = pd.read_parquet(path)
                keep_columns = ['inst','time','factor_value']
                df.drop(columns=set(df.columns)-set(keep_columns), inplace=True)
                df.rename(columns={'factor_value': factor_name + '_factor_value'},inplace=True)

                if count == 0:
                    total_df = df
                else:
                    total_df = pd.merge(total_df,df,on=['inst','time'],how='left')
                break
            
            
# step2: 标准化因子值：
# 这里使用MinMax还是Standard存疑，使用Standard因子统计显著性更好 
    scaler = MinMaxScaler()
    standlized_cols = []
    for factor_name in factor_lst:
        standlized_cols.append(factor_name + '_factor_value')

    # total_df[standlized_cols] = scaler.fit_transform(total_df[standlized_cols])
    # 按时间分组并标准化
    for time, group in total_df.groupby('time'):
        group[standlized_cols] = scaler.fit_transform(group[standlized_cols])
        total_df.loc[group.index, standlized_cols] = group[standlized_cols]


# step3: 计算每支股票每日的收益
    # 这里的circulation_a是流通市值的平方根    
    inst_return_df = pd.read_parquet('/home/linboyi/barrra-model/risk_factors-master/gen_return/data/result/factor_CTR_daily_none/cross-stock.parquet.gz')
    inst_return_df = inst_return_df[['inst','time','return','circulation_a']]
    total_df = pd.merge(total_df,inst_return_df,on=['inst','time'],how='left')

# step4: 数据预处理（处理nan）
    for factor in standlized_cols:
        # 用当日该因子的均值填充缺失值
        total_df[factor] = total_df.groupby('time')[factor].transform(lambda x: x.fillna(x.mean()))

    # 对于return和circulation_a同样
    total_df['return'] = total_df.groupby('time')['return'].transform(lambda x: x.fillna(x.mean()))
    total_df['circulation_a'] = total_df.groupby('time')['circulation_a'].transform(lambda x: x.fillna(x.mean()))

    # 再进行一遍清洗，处理某因子在某天所有股票都无数据
    for factor in standlized_cols:
        total_df[factor] = total_df[factor].fillna(total_df[factor].mean())



# step5: 回归求每日因子收益率

    factor_return = pd.DataFrame()

    for _, daily_data in total_df.groupby('time'):
        # 获取权重
        weights = daily_data['circulation_a']
        
        # 准备因子数据（自变量），这里因子列以 'factor_value' 结尾
        factors = daily_data.filter(regex='factor_value$')
        
        factors = sm.add_constant(factors)
        returns = daily_data['return']
        
        # 执行 WLS 回归
        model = sm.WLS(returns, factors, weights=weights)
        results = model.fit()
        coefficients = results.params
        
        # 将回归系数存储到结果 DataFrame 中
        factor_return = factor_return.append(coefficients, ignore_index=True)
        # factor_return['time'] = daily_data['time'].iloc[0]

    time_series = pd.DataFrame(total_df['time'].unique(), columns=['time'])
    factor_return['time'] = time_series['time']


# step6: 收益归因
    # 导入依据归因对象因子构建的投资组合的每日回报
    target_df = pd.read_csv('/home/linboyi/alpha_MEMO/data/result/backtest_MEMO/stock/1/asset.csv')
    target_df['time'] = target_df['trading_date']
    target_df['daily_return'] = target_df['pnl']/target_df['balance'].shift(1)
    target_df['daily_return'].iloc[0] = target_df['pnl'].iloc[0]/10000000
    combined_df = pd.merge(factor_return,target_df[['time','daily_return']],on='time',how='left')

    # 再做一次OLS，回归系数即为每个因子的贡献
    # 准备因子数据（自变量），这里因子列以 'factor_value' 结尾
    factors = combined_df.filter(regex='factor_value$')

    factors = sm.add_constant(factors)
    returns = combined_df['daily_return']

    # 执行 WLS 回归
    model = sm.OLS(returns, factors)
    results = model.fit()
    coefficients = results.params
    p_values = results.pvalues

    # 将回归系数存储到结果 DataFrame 中
    results_df = pd.DataFrame({'contribution': coefficients, 'p_values': p_values})
    results_df.sort_values(by = 'p_values', ascending=True,inplace=True)
    
    
    return results_df

def ploting(results_df):
# step7: 制图
    results_df.index = [idx.rstrip('_factor_value') for idx in results_df.index]
    # 创建一个柱状图
    plt.figure(figsize=(15, 8))  # 可以调整图形的大小
    bars = plt.bar(results_df.index, results_df['contribution'], color='skyblue')

    # 在每个柱形上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                ha='center', va='bottom')

    # 设置 X 轴标签
    plt.xticks(rotation=45)  # 旋转 X 轴标签，以便于阅读
    plt.xlabel('Factors')

    # 设置 Y 轴标签
    plt.ylabel('Contribution')

    # 设置图形标题
    plt.title('Contribution by Factors')

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    # 显示图形
    plt.savefig('/home/linboyi/barrra-model/risk_factors-master/return_attribution_MEMO.png')

if __name__ == '__main__':
    results_df = run_main()
    ploting(results_df)
