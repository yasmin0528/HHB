# milp_max_success_result1_1.py
# pip install pandas numpy pulp openpyxl

import pandas as pd
import numpy as np
import pulp


# ==== 数据预处理：统计每个站点最大小时借还车量 ====
TRIP_CSV = "D:/competition/code/202503-capitalbikeshare-tripdata.csv"
DEMAND_CSV = "D:/competition/code/demand_features.csv"
OUTPUT_XLSX = "D:/competition/code/result1_1.xlsx"

ALPHA = 1.0      # 借车成功的权重
BETA  = 1.0      # 还车成功的权重
MAX_CLOSE = 158
CAP_MULTIPLIER = 1.10
TOTAL_CAPACITY_NOW = 13670   # 现有总容量（请替换真实值）
Y_MIN = 0
Y_MAX_PER_STATION = 300
EPS = 1e-6       # 防0分母

# 读取原始数据
df_trip = pd.read_csv(TRIP_CSV, parse_dates=['started_at', 'ended_at'])

# 借车统计
df_trip['start_hour'] = df_trip['started_at'].dt.floor('H')
borrow_group = df_trip.groupby(['start_station_id', 'start_hour']).size().reset_index(name='borrow_count')
max_borrow = borrow_group.groupby('start_station_id')['borrow_count'].max().reset_index()

# 还车统计
df_trip['end_hour'] = df_trip['ended_at'].dt.floor('H')
return_group = df_trip.groupby(['end_station_id', 'end_hour']).size().reset_index(name='return_count')
max_return = return_group.groupby('end_station_id')['return_count'].max().reset_index()

# 合并
demand = pd.merge(max_borrow, max_return, left_on='start_station_id', right_on='end_station_id', how='outer')
demand['station_id'] = demand['start_station_id'].combine_first(demand['end_station_id'])
demand = demand[['station_id', 'borrow_count', 'return_count']].fillna(0)
demand.to_csv(DEMAND_CSV, index=False)

# ==== 读数据 ====
df = pd.read_csv(DEMAND_CSV)
need_cols = {"station_id","borrow_count","return_count"}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV 缺少列: {missing}")

df = df.copy()
df["borrow_count"] = df["borrow_count"].fillna(0).astype(float)
df["return_count"] = df["return_count"].fillna(0).astype(float)

stations = df["station_id"].tolist()
borrow = df["borrow_count"].to_numpy()
ret = df["return_count"].to_numpy()
n = len(stations)

# ==== 模型 ====
m = pulp.LpProblem("Maximize_Borrow_Return_Success", pulp.LpMaximize)

# 决策变量
x = pulp.LpVariable.dicts("keep", range(n), lowBound=0, upBound=1, cat="Binary")
y = pulp.LpVariable.dicts("cap",  range(n), lowBound=0, upBound=Y_MAX_PER_STATION, cat="Continuous")
s_b = pulp.LpVariable.dicts("sat_b", range(n), lowBound=0, upBound=None, cat="Continuous")
s_r = pulp.LpVariable.dicts("sat_r", range(n), lowBound=0, upBound=None, cat="Continuous")

# 目标：最大化加权成功率（线性）
m += pulp.lpSum([
    ALPHA * s_b[i] / max(borrow[i], EPS) + BETA * s_r[i] / max(ret[i], EPS)
    for i in range(n)
])

# 约束：最多关闭 158
m += pulp.lpSum([(1 - x[i]) for i in range(n)]) <= MAX_CLOSE

# 总容量上限
m += pulp.lpSum([y[i] for i in range(n)]) <= CAP_MULTIPLIER * float(TOTAL_CAPACITY_NOW)

# 耦合：容量只能给保留站
for i in range(n):
    m += y[i] <= Y_MAX_PER_STATION * x[i]
    # 新增约束：只要x[i]=1，则y[i]必须至少为1
    m += y[i] >= 1 * x[i]

# 线性化的 min 约束（被满足的借/还）
for i in range(n):
    # s_b ≤ y, ≤ borrow, ≤ M*x
    m += s_b[i] <= y[i]
    m += s_b[i] <= borrow[i]
    m += s_b[i] <= Y_MAX_PER_STATION * x[i]
    # s_r ≤ y, ≤ return, ≤ M*x
    m += s_r[i] <= y[i]
    m += s_r[i] <= ret[i]
    m += s_r[i] <= Y_MAX_PER_STATION * x[i]

# ==== 求解 ====
m.solve(pulp.PULP_CBC_CMD(msg=True))
print("Status:", pulp.LpStatus[m.status])
print("Objective:", pulp.value(m.objective))

keep = np.array([int(pulp.value(x[i]) > 0.5) for i in range(n)], dtype=int)
capacity = np.array([max(0, int(round(pulp.value(y[i])))) for i in range(n)], dtype=int)
capacity[keep == 0] = 0

out = pd.DataFrame({"station_id": stations, "keep": keep, "capacity": capacity})
print("保留数:", out["keep"].sum(), "总容量:", out["capacity"].sum())
out.to_excel(OUTPUT_XLSX, index=False)
print("已写出:", OUTPUT_XLSX)
total_borrow_success = sum([pulp.value(s_b[i]) for i in range(n)])
total_return_success = sum([pulp.value(s_r[i]) for i in range(n)])
borrow_success_rate = total_borrow_success / (borrow.sum() + EPS)
return_success_rate = total_return_success / (ret.sum() + EPS)
weighted_success = ALPHA * borrow_success_rate + BETA * return_success_rate

print(f"借车成功率: {borrow_success_rate:.4f}")
print(f"还车成功率: {return_success_rate:.4f}")
print(f"加权最大成功率: {weighted_success:.4f}")
