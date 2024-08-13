'''
单因子回测：
1. 必需实现 backtest_with_factor_func, backtest_with_factor_df , 保证两个函数能独立正常运行
2. 默认运行 backtest_with_factor_df
'''


import os
from typing import List
import pandas as pd
from zads import InstType, Frequency, symbols, Backtest, run_bts, CommonUtil,get_md


def backtest_with_factor_df(
    start:str = "20200101", # 行情起始日期
    end:str = "20220731", # 行情结束日期
    freq:Frequency = Frequency.DAILY, # 行情频率
    inst_types:List[InstType] = [InstType.STOCK], # 合约类型
    adjust:str = 'post', # 行情复权方式
    factor_freq:Frequency = Frequency.DAILY, # 因子频率
    factor_name:str = 'mega_score_values_100_factors_withComm', # 因子名称
    without_cache:bool = False,
    output_path:str = os.path.join(os.path.dirname(os.getcwd()),'data','result'), # 设置结果输出路径
    factor_file_name:str = 'mega_score_values_100_factors_corrected.csv',
    with_commission:bool = True
):

    '''
    回测：已有因子值，传入因子值以及因子名称等信息
    '''
    # output_path = os.path.join(os.path.dirname(os.getcwd()),'data','result') # 设置结果输出路径
    factor_file = os.path.join(output_path, factor_file_name)
    print(factor_file)
    factor_df = pd.read_csv(factor_file)#.rename(columns={'total_score':'factor_value'})

    # 回测相关参数
    factor_tminus = 1
    num_groups = 10 # 回测分组
    ascending = False # 因子值排序方式

    if with_commission:
        commissions_func = lambda q,p: abs(q * p * 0.00025) if(q>0) else abs(q * p * (0.00025 + 0.001))
    else:
        commissions_func = lambda q,p: abs(q * p * 0) if(q>0) else abs(q * p * 0)


    backtests = Backtest.get_backtest(
        start_dt=pd.Timestamp(start),
        end_dt=pd.Timestamp(end),
        freq=freq,
        inst_types=inst_types,
        adjust=adjust,
        factor_tminus=factor_tminus,
        factor_freq=factor_freq,
        factor_df=factor_df,
        factor_name=factor_name,
        num_groups=num_groups,
        ascending=ascending,
        path=output_path,
        commissions=commissions_func,
        without_cache=without_cache)


    run_bts(*backtests)



def run_bt_withComm_and_withoutComm(
    start:str = "20120101", # 行情起始日期
    end:str = "20220731", # 行情结束日期
    freq:Frequency = Frequency.DAILY, # 行情频率
    inst_types:List[InstType] = [InstType.STOCK], # 合约类型
    adjust:str = 'post', # 行情复权方式
    factor_freq:Frequency = Frequency.DAILY, # 因子频率
    factor_name:str = 'mega_score_values_100_factors', # 因子名称
    without_cache:bool = False,
    output_path:str = os.path.join(os.path.dirname(os.getcwd()),'data','result'), # 设置结果输出路径
    factor_file_name:str = 'mega_score_values_100_factors_corrected.csv'):

    # 费后
    backtest_with_factor_df(
        start = start, # 行情起始日期
        end = end, # 行情结束日期
        freq = freq, # 行情频率
        inst_types = inst_types, # 合约类型
        adjust = adjust, # 行情复权方式
        factor_freq =  factor_freq, # 因子频率
        factor_name = f"{factor_name}_withComm", # 因子名称
        without_cache = without_cache,
        output_path = output_path, # 设置结果输出路径
        factor_file_name = factor_file_name,
        with_commission = True)

    # 费前
    backtest_with_factor_df(
        start = start, # 行情起始日期
        end = end, # 行情结束日期
        freq = freq, # 行情频率
        inst_types = inst_types, # 合约类型
        adjust = adjust, # 行情复权方式
        factor_freq =  factor_freq, # 因子频率
        factor_name = f"{factor_name}_withoutComm", # 因子名称
        without_cache = without_cache,
        output_path = output_path, # 设置结果输出路径
        factor_file_name = factor_file_name,
        with_commission = False)



if __name__ == '__main__':

    barra_factor_list = [
        # "LNCAP",
        # "NLSIZE",
        "DASTD",
        "STOM",
        "RSTR"
        # "STOQ",
        # "STOA",
        # "CMRA",
        # "BETA",
        # "HSIGMA"
        ]
    start = "20120101"
    end = "20220731"

    # this_factor_name = "LNCAP"

    for this_factor_name in barra_factor_list:

        # 全时间段 monthly
        run_bt_withComm_and_withoutComm(
            start = start, # 行情起始日期
            end = end, # 行情结束日期
            freq = Frequency.DAILY, # 行情频率
            inst_types = [InstType.STOCK], # 合约类型
            adjust = 'post', # 行情复权方式
            factor_freq = Frequency.MONTHLY, # 因子频率
            factor_name = this_factor_name, # 因子名称
            without_cache = False,
            output_path = f"/home/chulin/Workspace/XianChulin/stock_multifactors/risk_factors/barra_{this_factor_name}/data/result/factor_{this_factor_name}_stock_daily_daily_post",
            factor_file_name = f"factor-{this_factor_name}-daily-stock-daily-post-{start}_{end}.csv")

