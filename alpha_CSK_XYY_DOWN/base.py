import pandas as pd
import numpy as np
from zads import FactorUtil, Frequency, Instrument, kline_day

# NOTICE: 此处说明因子引用的研报以及阐述其逻辑
# alpha_CSK_XYY_DOWN
# 来源 :  20240311-东北证券-因子选股系列之八：股票收益的协偏度因子
# 
# 逻辑 : 
#投资者通常对正偏（收益高于平均水平的可能性）有偏好，而对负偏（收益低于平均水平的可能性）有厌恶。这种风险偏好和赌博心理影响投资者对股票的定价。
#协偏度衡量个股收益相对于市场极端波动的敏感度。当市场出现极端上涨或下跌时，具有高协偏度的股票收益会与市场波动更加同步，表现出更大的波动性。
#由于投资者偏好正偏而厌恶负偏，低协偏度的股票（即在市场极端波动时表现相对稳定的个股）预期会有更高的回报。这种预期回报的差异导致了协偏度溢价的存在。
#因此，因子整体和未来个股收益呈负相关关系。
#投资者大多属于风险厌恶型，因此在市场出现极端波动时，那些产生了较大损失的股票不受投资者偏好，这会造成此类高损失风险的股票相对于与其内在价值被低估，
#于是产生了负协偏度异象。但事实上，在极端下跌市场和极端上涨市场中，投资者的风险偏好和承受能力是不同的。
#历史上市场发生极端上行风险时上涨的股票，未来收益相对较优；历史上市场发生极端下行风险时，下跌较多的股票的未来表现较优。
#因此，上行协偏度因子和个股收益呈正相关。下行协偏度因子和个股收益呈负相关。

#计算个股日收益率
def calculate_stock_returns(inst_df_day:pd.DataFrame):
   # 剔除ST、ST*股票
    #if 'name' in inst_df_day.columns:  # 假设DataFrame中有一列名为'name'，包含股票名称
    #    st_mask = (inst_df_day['name'].str.contains('ST')) | (inst_df_day['name'].str.contains('ST*'))
    #    inst_df_day = inst_df_day[~st_mask]

 # 检查交易量是否为0，如果是，则标记为停牌
    inst_df_day['is_suspended'] = inst_df_day['volume'] == 0

    inst_df_day.sort_index(inplace=True)

   # 剔除上市不足一年的股票（待实现功能）
    #if 'list_date' in inst_df_day.columns:  # 假设DataFrame中有一列名为'list_date'，包含上市日期
    #    today = pd.to_datetime(date)
    #    listed_one_year_ago = today - pd.Timedelta(days=365)
    #    mask = inst_df_day_date['list_date'] >= listed_one_year_ago
    #    inst_df_day_date = inst_df_day_date[mask]

    #计算个股简单日收益率（可尝试对数收益率）
    stock_returns = inst_df_day['close'].pct_change()

    stock_returns[inst_df_day['is_suspended']] = np.nan

    return stock_returns


#使用中证全指（000985.CSI）的日频收益率作为市场收益的代表
def calculate_market_returns(inst_df_day: pd.DataFrame):

    start = inst_df_day['time'][0]
    end = inst_df_day['time'][-1]
    #传入中证全指指数
    csi_df = kline_day('000985',start,end)

    # 确保传入的DataFrame索引是按日期排序的
    csi_df.sort_index(inplace=True)
    
    # 剔除上市不足一年的股票（如果数据中包含上市日期）（待实现功能）
    #if 'list_date' in index_df.columns:
     #   today = pd.to_datetime(date)  # 确保date是pd.Timestamp类型
      #  listed_one_year_ago = today - pd.Timedelta(days=365)
       # mask = index_df['list_date'] >= listed_one_year_ago
        #index_df = index_df.loc[mask]  # 使用.loc来选择满足条件的行
    
    # 计算简单收益率
    market_returns = csi_df['close'].pct_change()
       
    # 返回特定日期的市场收益率
    # 确保date是函数参数，用于索引特定日期的市场收益率
    return market_returns



#计算下行协偏度因子,回溯窗口默认为最优的120D，其他参考时长可以看研报
def calculate_CSK_XYY_DOWN(inst_df_day: pd.DataFrame, rolling_window: int = 120):
    inst_df_day = inst_df_day.sort_index()
    inst_df_day['stock_returns'] = calculate_stock_returns(inst_df_day)
    inst_df_day['market_returns'] = calculate_market_returns(inst_df_day)

    #个股收益均值和标准差
    stock_returns_mean = inst_df_day['stock_returns'].rolling(rolling_window).mean()
    market_returns_mean = inst_df_day['market_returns'].rolling(rolling_window).mean()
    stock_returns_std = inst_df_day['stock_returns'].rolling(rolling_window).std()
    market_returns_std = inst_df_day['market_returns'].rolling(rolling_window).std()

    #标准化收益
    stock_normalized_returns = (inst_df_day['stock_returns'] - stock_returns_mean) / stock_returns_std
    market_normalized_returns = (inst_df_day['market_returns'] - market_returns_mean) / market_returns_std

    # 标记下行市场的个股
    down_market_mask = inst_df_day['market_returns'] < market_returns_mean

    # 计算下行协偏度因子
    csk_down = [np.nan] * len(inst_df_day)
    for i in range(rolling_window, len(inst_df_day)):
        if down_market_mask[i-rolling_window:i].sum() > 1:
            X = stock_normalized_returns[i-rolling_window:i][down_market_mask[i-rolling_window:i]]
            Y = market_normalized_returns[i-rolling_window:i][down_market_mask[i-rolling_window:i]]
            X_mean = X.mean()
            Y_mean = Y.mean()

            numerator = ((X - X_mean) * (Y - Y_mean) ** 2).mean()
            denominator = np.sqrt(((X - X_mean) ** 2).mean() * ((Y - Y_mean) ** 2).mean())
            
            # 检查分母是否为0
            if denominator != 0:
                csk_down[i] = numerator / denominator
            else:
                csk_down[i] = np.nan  # Handle zero denominator case

    inst_df_day['CSK_XYY_DOWN'] = csk_down
    return inst_df_day



# NOTICE: 此处实现单合约的时序因子逻辑
# 1. 第1个参数必需定义：inst_obj:Instrument
# 2. 其他参数可自由按需定义, 但必须给定默认值
# 3. 返回值将会被用于 reduce (对合约的时序因子进行横截面处理) 
def CSK_XYY_DOWN_by_inst(inst_obj, rolling_window: int = 120):
    inst_code = inst_obj.code
    daily_df = inst_obj.kline_day

    try:
        daily_df = calculate_CSK_XYY_DOWN(daily_df, rolling_window=rolling_window)
        daily_df['inst'] = inst_code

        if 'CSK_XYY_DOWN' not in daily_df.columns:
            raise KeyError(f"Column 'CSK_XYY_DOWN' not found in the DataFrame for inst {inst_code}")

        return daily_df[['inst', 'time', 'volume', 'CSK_XYY_DOWN']]
    except Exception as e:
        print(f"Error processing inst {inst_code}: {e}")
        return None


# NOTICE: 对合约的时序因子进行横截面处理
# 1. 前2个参数必需定义, 且顺序不可变更: factor_df, freq
# 2. 其他参数可自由按需定义, 但必须给定默认值
# 3. 做横截面处理, 需注意处理方法的顺序, 严格按照因子逻辑进行处理
def CSK_XYY_DOWN(
    factor_df:pd.DataFrame,
    freq:Frequency=Frequency.DAILY):

    # 横截面z-score标准化
    factor_df = FactorUtil.crssct_stddzt(
        factor_df, 
        ['CSK_XYY_DOWN']) 

    #     
    #     将停牌时的因子值设为NA
    factor_df['CSK_XYY_DOWN'] = np.where(factor_df['volume']<=0, np.NaN, factor_df['CSK_XYY_DOWN'])

    # !!! NOTICE: 因子重采样
    #     适用场景: 高频转低频
    #     譬如 factor_df 为日频因子, 但是最终想输出月频因子
    if(freq>Frequency.DAILY):
        factor_df = FactorUtil.resample(factor_df=factor_df, freq=freq)

    # 横截面 MAD 去极值（待实现功能）
    #factor_df = FactorUtil.crssct_MAD(factor_df, ['CSK_XYY_DOWN'])

    factor_df.rename(columns={'CSK_XYY_DOWN':'factor_value'}, inplace=True)
    return factor_df[['inst','time','factor_value']]
