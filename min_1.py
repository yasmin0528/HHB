import time
import numpy as np
import pandas as pd
import cvxpy as cp

# ===== 参数配置 =====
DEMAND_CSV = "demand_features.csv"
OUTPUT_XLSX = "result1_2.xlsx"

ALPHA = 1.0
BETA = 1.0
MAX_CLOSE = 158
CAP_MULTIPLIER = 1.10
TOTAL_CAPACITY_NOW = 13670
Y_MIN = 1
Y_MAX_PER_STATION = 200

# ===== 读取数据 =====
print("读取数据...")
df = pd.read_csv(DEMAND_CSV)
df["borrow_count"] = df["borrow_count"].fillna(0).clip(lower=0)
df["return_count"] = df["return_count"].fillna(0).clip(lower=0)
df["predicted_demand"] = df["predicted_demand"].fillna(0).clip(lower=0)
df["imbalance"] = (df["borrow_count"] - df["return_count"]).abs()
df["predicted_daily"] = df["predicted_demand"] / 32.0 

n = len(df)
imbalance = df["imbalance"].values
pred_daily = df["predicted_daily"].values

# ===== 连续松弛求解 =====
print("开始连续松弛求解...")

x = cp.Variable(n)
y = cp.Variable(n)

scheduling_cost = ALPHA * cp.sum(cp.multiply(imbalance, x))
vacancy_loss = BETA * cp.sum_squares(y - pred_daily)
objective = cp.Minimize(scheduling_cost + vacancy_loss)

constraints = [
    cp.sum(1 - x) <= MAX_CLOSE,
    cp.sum(y) <= CAP_MULTIPLIER * TOTAL_CAPACITY_NOW,
    y <= Y_MAX_PER_STATION * x,
    y >= Y_MIN * x,
    x >= 0,
    x <= 1,
    y >= 0,
]

prob = cp.Problem(objective, constraints)
start_time = time.time()
prob.solve(solver=cp.ECOS, verbose=True)
print(f"连续松弛求解完成，用时 {time.time() - start_time:.2f} 秒，状态 {prob.status}")

# ===== 严格控制关闭数，四舍五入 =====
n_keep = n - MAX_CLOSE
idx_sorted = np.argsort(-x.value)  # 按x值降序排序
keep_idx = idx_sorted[:n_keep]
close_idx = idx_sorted[n_keep:]

x_val = np.zeros(n, dtype=int)
x_val[keep_idx] = 1
x_val[close_idx] = 0

y_val = y.value.copy()
y_val[close_idx] = 0

total_cap = y_val.sum()
max_cap = CAP_MULTIPLIER * TOTAL_CAPACITY_NOW

if total_cap > max_cap:
    scale = max_cap / total_cap
    y_val[keep_idx] = np.clip(y_val[keep_idx] * scale, Y_MIN, Y_MAX_PER_STATION)

y_val[keep_idx] = np.clip(y_val[keep_idx], Y_MIN, Y_MAX_PER_STATION)
y_val[close_idx] = 0
y_val = np.rint(y_val).astype(int)

print(f"初步解：保留 {x_val.sum()} 站点，关闭 {n - x_val.sum()} 站点，总容量 {y_val.sum()}")

# ===== 局部搜索优化 =====
def local_search_optimize(df, x_init, y_init, alpha, beta, max_close, cap_multiplier, total_capacity_now, y_min, y_max, max_iter=500, no_improve_limit=50):
    n = len(df)
    imbalance = df["imbalance"].values
    pred_daily = df["predicted_demand"].values / 32.0 
    
    x_best = x_init.copy()
    y_best = y_init.copy()
    
    def calc_obj(x, y):
        sched = alpha * np.sum(imbalance * x)
        vac_loss = beta * np.sum((y - pred_daily) ** 2)
        return sched + vac_loss
    
    best_obj = calc_obj(x_best, y_best)
    print(f"局部搜索初始目标值: {best_obj:.4f}")
    
    no_improve_count = 0
    
    for it in range(max_iter):
        improved = False
        open_idx = np.where(x_best == 1)[0]
        close_idx = np.where(x_best == 0)[0]
        
        np.random.shuffle(open_idx)
        np.random.shuffle(close_idx)
        
        try_num = min(20, len(open_idx), len(close_idx))
        
        for i in range(try_num):
            o = open_idx[i]
            c = close_idx[i]
            
            x_new = x_best.copy()
            x_new[o] = 0
            x_new[c] = 1
            
            y_new = y_best.copy()
            y_new[o] = 0
            y_new[c] = max(y_min, min(y_max, pred_daily[c]))
            
            kept_idx = np.where(x_new == 1)[0]
            total_cap = y_new[kept_idx].sum()
            cap_limit = cap_multiplier * total_capacity_now
            
            if total_cap > cap_limit:
                scale = cap_limit / total_cap
                y_new[kept_idx] = np.clip(y_new[kept_idx] * scale, y_min, y_max)
            
            obj_new = calc_obj(x_new, y_new)
            
            if obj_new < best_obj:
                x_best = x_new
                y_best = y_new
                best_obj = obj_new
                improved = True
                print(f"第{it+1}次迭代改进: 目标值降至 {best_obj:.4f}")
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= no_improve_limit:
                print(f"连续{no_improve_limit}次无改进，停止搜索。")
                break
        else:
            no_improve_count = 0
    
    return x_best, y_best, best_obj

print("开始局部搜索优化...")
start_ls = time.time()
x_opt, y_opt, obj_opt = local_search_optimize(df, x_val, y_val, ALPHA, BETA, MAX_CLOSE, CAP_MULTIPLIER, TOTAL_CAPACITY_NOW, Y_MIN, Y_MAX_PER_STATION)
print(f"局部搜索完成，用时 {time.time() - start_ls:.2f} 秒，最终目标值 {obj_opt:.4f}")

# 保存最终结果
result_df = pd.DataFrame({
    "station_id": df["station_id"],
    "keep": x_opt,
    "capacity": np.rint(y_opt).astype(int)
})

print(f"最终保留站点数: {x_opt.sum()}，关闭站点数: {n - x_opt.sum()}")
print(f"最终总容量: {result_df['capacity'].sum()}")

result_df.to_excel(OUTPUT_XLSX, index=False)
print(f"结果保存至 {OUTPUT_XLSX}")
