import pandas as pd

# 读取官方数据集
loc = pd.read_csv(r'D:\competition\code\停车位推算\Capital_Bikeshare_Locations.csv')

# 读取25年3月站点名（已预处理，假设文件已存在）
stations = pd.read_excel(r'D:\competition\code\停车位推算\station_id_name_202503.xlsx')

# 标准化站点名
loc['station_name_std'] = loc['NAME'].str.strip().str.upper()
stations['station_name_std'] = stations['station_name'].str.strip().str.upper()

# 合并，提取有对应站点名的数据
merged = pd.merge(stations, loc, on='station_name_std', how='inner')

# 计算总停车位数
merged['docks_total'] = merged['CAPACITY']

# 输出数量检查
print('匹配站点数量：', len(merged))

# 保存结果
merged[['station_id', 'station_name', 'docks_total']].to_csv(r'D:\competition\code\停车位推算\official_docks_202503.csv', index=False, encoding='utf-8-sig')
