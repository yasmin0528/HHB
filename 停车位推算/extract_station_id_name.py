import pandas as pd

# 读取数据
df = pd.read_csv(r'D:\competition\code\202503-capitalbikeshare-tripdata.csv')

# 提取所有起点和终点的站点信息
start = df[['start_station_id', 'start_station_name']].drop_duplicates()
end = df[['end_station_id', 'end_station_name']].drop_duplicates()

# 统一字段名
start = start.rename(columns={'start_station_id': 'station_id', 'start_station_name': 'station_name'})
end = end.rename(columns={'end_station_id': 'station_id', 'end_station_name': 'station_name'})

# 合并并去重
stations = pd.concat([start, end], ignore_index=True).drop_duplicates(subset='station_id')

# 按id升序排列
stations = stations.sort_values(by='station_id').reset_index(drop=True)

# 输出数量检查
print("站点数量：", len(stations))  # 应为793

# 保存结果
stations.to_csv(r'D:\competition\code\station_id_name_202503.csv', index=False, encoding='utf-8-sig')
