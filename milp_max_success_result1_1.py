
# milp_max_success_result1_1.py
# pip install pandas numpy pulp openpyxl

import pandas as pd
import numpy as np
import pulp

# ==== 数据预处理：净车数曲线法统计每站点所需容量 ====
TRIP_CSV = "D:/competition/code/202503-capitalbikeshare-tripdata.csv"
OUTPUT_XLSX = "D:/competition/code/result1_1.xlsx"
TOTAL_CAPACITY_NOW = 29742   # 现有总容量（请替换真实值）
Y_MAX_PER_STATION = 300

# 读取原始数据
df_trip = pd.read_csv(TRIP_CSV, parse_dates=['started_at', 'ended_at'])

# 构建事件流
borrow_events = pd.DataFrame({
    "event_time": df_trip["started_at"].values,
    "station": df_trip["start_station_id"].values,
    "delta": -1
})
return_events = pd.DataFrame({
    "event_time": df_trip["ended_at"].values,
    "station": df_trip["end_station_id"].values,
    "delta": 1
})
events = pd.concat([borrow_events, return_events], ignore_index=True)
events = events.dropna(subset=["event_time", "station"]).copy()
events = events.sort_values("event_time", kind="mergesort").reset_index(drop=True)
events["cum_change"] = events.groupby("station")["delta"].cumsum()

# 统计每站点最大净变化区间
agg = events.groupby("station")["cum_change"].agg(["max", "min"]).rename(
    columns={"max": "Max_Positive", "min": "Min_Negative"}
)
agg["Required_Docks_Est"] = agg["Max_Positive"] - agg["Min_Negative"]
agg = agg.reset_index()
stations = agg["station"].tolist()
demand = agg["Required_Docks_Est"].to_numpy()
n = len(stations)

# ==== 优化分配停车位容量 ====
m = pulp.LpProblem("Minimize_Lack_Capacity", pulp.LpMinimize)
y = pulp.LpVariable.dicts("cap", range(n), lowBound=0, upBound=Y_MAX_PER_STATION, cat="Integer")
lack = pulp.LpVariable.dicts("lack", range(n), lowBound=0, upBound=None, cat="Continuous")

# 目标：最小化分配不足
m += pulp.lpSum([lack[i] for i in range(n)])

# 约束
for i in range(n):
    m += lack[i] >= demand[i] - y[i]
    m += lack[i] >= 0
m += pulp.lpSum([y[i] for i in range(n)]) <= TOTAL_CAPACITY_NOW

# ==== 求解与输出 ====
m.solve(pulp.PULP_CBC_CMD(msg=True))
print("Status:", pulp.LpStatus[m.status])
print("Objective (总分配不足):", pulp.value(m.objective))

capacity = np.array([max(0, int(round(pulp.value(y[i])))) for i in range(n)], dtype=int)
lack_val = np.array([max(0, pulp.value(lack[i])) for i in range(n)], dtype=int)
# status: 1 表示分配满足需求，0 表示有缺口
status = np.array([1 if lack_val[i]==0 else 0 for i in range(n)], dtype=int)

# 输出表头为 station_id、status、capacity
out = pd.DataFrame({
    "station_id": stations,
    "status": status,
    "capacity": capacity
})
print("总分配容量:", out["capacity"].sum())
print("满足需求站点数:", out["status"].sum())
out.to_excel(OUTPUT_XLSX, index=False)
print("已写出:", OUTPUT_XLSX)
# 近似借还成功率统计
total_stations = len(out)
success_stations = int(out["status"].sum())
success_rate = success_stations / total_stations if total_stations > 0 else 0
print(f"借还成功率（满足需求站点比例）: {success_rate:.4f}")
