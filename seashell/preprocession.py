from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('initial_data.csv')

result = df[['楼层', '住房面积', '价格（元/月）']]
result.loc[:, '楼层']= result.loc[:, '楼层'].str.findall(r'\d+') # 提取数字
result.loc[:, '楼层'] =result.loc[:, '楼层'].str.join('').str.replace(r'[\[\]]', '', regex=True).astype(int) # 去中括号

result.loc[:, '住房面积'] = result.loc[:, '住房面积'].str.findall(r'\d+\.?\d*')
result.loc[:, '住房面积'] = result.loc[:, '住房面积'].str.join('').str.replace(r'[\[\]]', '', regex=True).astype(float)
result.loc[:, '价格（元/月）'] =result.loc[:, '价格（元/月）'].astype(int)
df[['楼层', '住房面积', '价格（元/月）']] = result

scaler = StandardScaler()
data_scaled = scaler.fit_transform(result)
data_scaled = pd.DataFrame(data_scaled, columns=result.columns)











