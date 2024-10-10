import pandas as pd
import numpy as np
from data import df

def check_length(lst):
    return len(lst) >= 5

def add_nan_if_needed(lst):
    if len(lst) < 6:
        lst.insert(0, np.nan)  # 在列表首位添加NaN
    return lst

# 数据包含区域值，计算'-'前后数值的平均值
def replace_with_average(s):
    if '-' in s:
        parts = s.split('-') # 分割字符串，并确保分割后有两个部分
        if len(parts) == 2:
            try:
                value1 = int(parts[0])
                value2 = int(parts[1])
                return (value1 + value2) / 2 # 计算平均值
            except ValueError:
                return s
    return s

df = df[df.loc[:, '属性'].apply(check_length)] # 筛选列表正常的数据
df.loc[:, '价格（元/月）'] = df.loc[:, '价格（元/月）'].apply(replace_with_average)

# 应用该函数到'属性'列
df.loc[:, '属性'] = df.loc[:, '属性'].apply(add_nan_if_needed)

# 创建一个新的DataFrame，其中包含拆分后的列表数据
new_columns = ['是否精选', '地址', '住房面积', '房子朝向', '户型', '楼层']  # 创建新的列名
expanded_df = df.loc[:, '属性'].apply(lambda x: pd.Series(x, index=new_columns))

# 将新的DataFrame与原始DataFrame合并
df = pd.concat([df.drop('属性', axis=1), expanded_df], axis=1)

new_col= ['地址', '房子朝向', '租贷类型', '户型', '楼层', '住房面积', '价格（元/月）', '来源', '特点', '是否精选']
df = df.reindex(columns=new_col)

def conversion(value):
    if value == '精选':
        return 1
    else:
        return 0
df['是否精选'].str.replace(' ', '')
df['是否精选'] = df['是否精选'].apply(conversion)
df.fillna(0, inplace=True)
df['是否精选'].astype(int)
df['是否精选'].unique()

file_path = 'initial_data.csv'
df.to_csv(file_path, index=False, encoding='utf_8_sig')