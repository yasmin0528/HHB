# pareto_capacity_allocation_with_real_event_sim.py
"""
Pareto 权重法 + 事件流真实成功率模拟（完整、稳健版）
保存：每个权重的分配文件 result_w{w:.2f}.xlsx 和 pareto_summary.csv
"""

import time
import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

# ========== 配置 ==========
INPUT_CSV = "demand_features.csv"
RIDES_CSV = "202503-capitalbikeshare-tripdata.csv"
OUTPUT_DIR = Path("pareto_results")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CLOSE = 158
CAP_MULTIPLIER = 1.10
TOTAL_CAPACITY_NOW = 33104
Y_MIN = 1
Y_MAX_PER_STATION = 600
ALPHA = 1.0
BETA = 1.0

# 权重扫描（0..1）
WEIGHTS = np.linspace(0.0, 1.0, 21)

# 事件模拟参数
INITIAL_FILL_FRACTION = 1.0  # 初始库存 = capacity * INITIAL_FILL_FRACTION

# ========== 读取需求数据 ==========
print("读取需求数据:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, dtype={})
required_cols = {"station_id", "borrow_count", "return_count", "predicted_demand"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV 缺少必须列: {missing}")

# 规范化 station_id 为字符串（避免类型不一致）
df["station_id"] = df["station_id"].astype(str)

df["borrow_count"] = df["borrow_count"].fillna(0).astype(float).clip(lower=0)
df["return_count"] = df["return_count"].fillna(0).astype(float).clip(lower=0)
df["predicted_demand"] = df["predicted_demand"].fillna(0).astype(float).clip(lower=0)
df["imbalance"] = (df["borrow_count"] - df["return_count"]).abs().astype(float)
df["pred_daily"] = df["predicted_demand"] / 32.0

stations = df["station_id"].tolist()
n = len(stations)
n_keep = n - MAX_CLOSE

# required: 优先使用 Required_Docks_Est，否则使用 pred_daily
if "Required_Docks_Est" in df.columns:
    required = df["Required_Docks_Est"].astype(float).values
else:
    required = df["pred_daily"].astype(float).values

imbalance = df["imbalance"].values.astype(float)
pred_daily = df["pred_daily"].values.astype(float)

# ========== 基准值（归一化） ==========
total_required = required.sum() + 1e-9
x_ref = np.ones(n)
y_ref = pred_daily.copy()
baseline_op_cost = (ALPHA * np.sum(imbalance * x_ref) +
                    BETA * np.sum((pred_daily - y_ref) ** 2)) + 1e-9

# ========== 读取并预处理骑行明细 ==========
print("读取骑行明细:", RIDES_CSV)
rides_df = pd.read_csv(RIDES_CSV, dtype=str)  # 先读为字符串，后统一处理
# 必要列检查
needed_cols = {"start_station_id", "end_station_id", "started_at", "ended_at"}
if not needed_cols.issubset(rides_df.columns):
    raise ValueError(f"{RIDES_CSV} 必须包含列: {needed_cols}")

# 将站点 id 也转成字符串，和 df 中一致
rides_df["start_station_id"] = rides_df["start_station_id"].astype(str)
rides_df["end_station_id"] = rides_df["end_station_id"].astype(str)

def parse_time_series(series):
    """尝试把时间列解析成 pd.Timestamp 或相对秒（float）用于排序。
       支持标准时间字符串，也支持 mm:ss.s 或 hh:mm:ss 等格式；若解析失败则返回 NaT."""
    # 先尝试 pandas to_datetime (常见情况)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() > 0.9 * len(series):  # 大多数能解析
        return parsed

    # 否则尝试解析像 "MM:SS.s" 或 "HH:MM:SS" -> 解析为 timedelta from day 0
    def parse_hms(s):
        s = str(s).strip()
        if s == "" or s.lower() == "nan":
            return pd.NaT
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except:
            return pd.NaT
        # 从右到左： seconds, minutes, hours
        if len(parts) == 1:
            secs = parts[0]
        elif len(parts) == 2:
            mins, secs = parts
            secs = mins * 60.0 + secs
        elif len(parts) == 3:
            h, m, s2 = parts
            secs = h * 3600.0 + m * 60.0 + s2
        else:
            return pd.NaT
        # convert to Timestamp anchored at 1970-01-01
        return pd.Timestamp("1970-01-01") + pd.to_timedelta(secs, unit="s")

    parsed2 = series.map(parse_hms)
    # 如果仍然多为 NaT, fallback: try numeric (seconds)
    if parsed2.notna().sum() >= 1:
        return pd.to_datetime(parsed2)
    # 否则 final fallback: try numeric float seconds
    def try_float(s):
        try:
            return float(s)
        except:
            return np.nan
    numeric = series.map(try_float)
    if numeric.notna().sum() > 0:
        # anchor as timestamps from day 1970-01-01
        return pd.to_datetime(pd.to_timedelta(numeric.fillna(0), unit="s")) + pd.Timestamp("1970-01-01")
    # 全失败
    return pd.Series([pd.NaT]*len(series))

# 解析 started_at / ended_at
rides_df["started_at_parsed"] = parse_time_series(rides_df["started_at"])
rides_df["ended_at_parsed"] = parse_time_series(rides_df["ended_at"])

# 丢弃无法解析时间的记录（并警告）
before = len(rides_df)
rides_df = rides_df.dropna(subset=["started_at_parsed", "ended_at_parsed"]).reset_index(drop=True)
dropped = before - len(rides_df)
if dropped > 0:
    print(f"⚠️ 丢弃 {dropped} 条无法解析时间的骑行记录")

# ========== 工具函数 ==========
def compute_operational_cost(x_vec, y_vec, alpha=ALPHA, beta=BETA):
    sched = alpha * float(np.sum(imbalance * x_vec))
    vac = beta * float(np.sum((pred_daily - y_vec) ** 2))
    return sched + vac, sched, vac

def compute_success_rate(y_vec, required_arr):
    mask = required_arr > 1e-9
    ratio = np.ones_like(required_arr, dtype=float)
    ratio[mask] = np.clip(y_vec[mask] / required_arr[mask], 0.0, 1.0)
    avg_success = float(ratio.mean())
    success_count = int(np.sum(y_vec >= required_arr - 1e-9))
    return avg_success, success_count

def compute_total_lack(y_vec, required_arr):
    lack = np.maximum(0.0, required_arr - y_vec)
    return float(lack.sum())

def simulate_event_level_success(capacity_vec, keep_vec, rides_events_df, stations_list, initial_fill_frac=1.0):
    """
    事件级模拟：
      - capacity_vec: 每个站点分配容量（按 stations_list 顺序）
      - keep_vec: 0/1 表示保留或关闭（按 stations_list 顺序）
      - rides_events_df: 包含 started_at_parsed, ended_at_parsed, start_station_id, end_station_id
    返回：事件级成功率（borrow+return 成功事件 / 总事件）
    """
    # map station id -> index
    station_to_idx = {sid: i for i, sid in enumerate(stations_list)}
    # 初始库存：capacity * fraction（四舍五入）
    capacity_arr = np.array(capacity_vec, dtype=float)
    inventory = np.floor(capacity_arr * float(initial_fill_frac)).astype(int)
    capacity_limit = capacity_arr.astype(int)

    # 构造事件流 DataFrame: station_id, time, change (+1 for return, -1 for borrow), event_type
    bor = rides_events_df[["start_station_id", "started_at_parsed"]].rename(
        columns={"start_station_id": "station_id", "started_at_parsed": "time"})
    bor = bor.assign(change=-1, event_type="borrow")
    ret = rides_events_df[["end_station_id", "ended_at_parsed"]].rename(
        columns={"end_station_id": "station_id", "ended_at_parsed": "time"})
    ret = ret.assign(change=+1, event_type="return")

    events = pd.concat([bor, ret], ignore_index=True)
    # 转 station_id 为字符串以匹配
    events["station_id"] = events["station_id"].astype(str)

    # 排序：按时间，若同一时间同时有 return 和 borrow，先处理 return（释放空位），所以 change descending
    events = events.sort_values(by=["time", "change"], ascending=[True, False]).reset_index(drop=True)

    success = 0
    total = 0

    for _, row in events.iterrows():
        sid = str(row["station_id"])
        if sid not in station_to_idx:
            # 不在我们分配表中的站点，直接算为失败（或可跳过）
            total += 1
            continue
        idx = station_to_idx[sid]
        if keep_vec[idx] == 0:
            # 站点关闭 -> 无论借还都失败
            total += 1
            continue
        if row["change"] == -1:
            # borrow
            total += 1
            if inventory[idx] > 0:
                inventory[idx] -= 1
                success += 1
        else:
            # return
            total += 1
            if inventory[idx] < capacity_limit[idx]:
                inventory[idx] += 1
                success += 1
    return success / total if total > 0 else 0.0

# ========== 主循环：遍历权重，求解并模拟 ==========
summary_rows = []

for w in WEIGHTS:
    print(f"\n=== 求解权重 w = {w:.3f} ===")
    t0 = time.time()

    x = cp.Variable(n)   # 连续 relax
    y = cp.Variable(n)
    lack = cp.Variable(n, nonneg=True)

    scheduling_cost = cp.sum(cp.multiply(imbalance, x)) * ALPHA
    vacancy_loss = BETA * cp.sum_squares(y - pred_daily)
    op_cost_expr = (scheduling_cost + vacancy_loss) / baseline_op_cost

    lack_constraints = [lack >= required - y, lack >= 0]
    lack_sum_expr = cp.sum(lack) / total_required

    objective = cp.Minimize(w * op_cost_expr + (1.0 - w) * lack_sum_expr)

    cons = lack_constraints + [
        cp.sum(1 - x) <= MAX_CLOSE,
        cp.sum(y) <= CAP_MULTIPLIER * TOTAL_CAPACITY_NOW,
        y <= Y_MAX_PER_STATION * x,
        y >= Y_MIN * x,
        x >= 0, x <= 1,
        y >= 0
    ]

    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print("求解失败，尝试默认求解器：", e)
        try:
            prob.solve(verbose=False)
        except Exception as e2:
            print("求解器全部失败，跳过该权重：", e2)
            continue

    # 获取松弛解并做严格保留截断
    x_rel = np.clip(np.nan_to_num(x.value, nan=0.0), 0.0, 1.0)
    y_rel = np.nan_to_num(y.value, nan=0.0)

    idx_sorted = np.argsort(-x_rel)
    keep_idx = idx_sorted[:n_keep]
    close_idx = idx_sorted[n_keep:]

    x_round = np.zeros(n, dtype=int)
    x_round[keep_idx] = 1

    y_round = y_rel.copy()
    y_round[close_idx] = 0.0

    # 确保总容量上限
    kept_idx = np.where(x_round == 1)[0]
    cap_limit = CAP_MULTIPLIER * TOTAL_CAPACITY_NOW
    total_cap = float(y_round[kept_idx].sum()) if kept_idx.size > 0 else 0.0
    if total_cap > cap_limit and total_cap > 0:
        scale = cap_limit / total_cap
        y_round[kept_idx] = np.clip(y_round[kept_idx] * scale, Y_MIN, Y_MAX_PER_STATION)

    # 强制上下界并四舍五入
    y_round[kept_idx] = np.clip(y_round[kept_idx], Y_MIN, Y_MAX_PER_STATION)
    y_round = np.rint(y_round).astype(int)
    y_round[close_idx] = 0

    x_final = x_round
    y_final = y_round

    # 计算预测/平滑指标
    op_cost, sched_val, vac_val = compute_operational_cost(x_final, y_final)
    total_lack = compute_total_lack(y_final, required)
    pred_success_rate, success_count = compute_success_rate(y_final, required)

    # 事件流动态仿真（真实成功率）
    real_success_rate = simulate_event_level_success(y_final, x_final, rides_df,
                                                     stations, initial_fill_frac=INITIAL_FILL_FRACTION)

    t_elapsed = time.time() - t0
    print(f"w={w:.3f}  op_cost={op_cost:.2f}  lack={total_lack:.2f}  "
          f"pred_success={pred_success_rate:.3f}  real_success={real_success_rate:.3f}  time={t_elapsed:.1f}s")

    # 保存 summary
    summary_rows.append({
        "w": float(w),
        "op_cost": float(op_cost),
        "scheduling": float(sched_val),
        "vacancy": float(vac_val),
        "total_lack": float(total_lack),
        "pred_success_rate": float(pred_success_rate),
        "success_count": int(success_count),
        "real_success_rate": float(real_success_rate),
        "time_s": float(t_elapsed)
    })

    # 保存分配表
    res_df = pd.DataFrame({
        "station_id": df["station_id"].values,
        "keep": x_final,
        "capacity": y_final
    })
    out_file = OUTPUT_DIR / f"result_w{w:.2f}.xlsx"
    res_df.to_excel(out_file, index=False)

# 最后保存 summary 与绘图
summary_df = pd.DataFrame(summary_rows).sort_values("w").reset_index(drop=True)
summary_df.to_csv(OUTPUT_DIR / "pareto_summary.csv", index=False)
print("Pareto 汇总已保存：", OUTPUT_DIR / "pareto_summary.csv")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(summary_df["op_cost"], summary_df["pred_success_rate"], "-o", label="Predicted (smooth)")
    plt.plot(summary_df["op_cost"], summary_df["real_success_rate"], "-s", label="Real (event-sim)")
    for _, row in summary_df.iterrows():
        plt.text(row["op_cost"], row["real_success_rate"], f"w={row['w']:.2f}", fontsize=8)
    plt.xlabel("Operational Cost")
    plt.ylabel("Success Rate")
    plt.title("Pareto Frontier: Predicted vs Real")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pareto_frontier.png", dpi=300)
    print("Pareto 图已保存")
except Exception as e:
    print("绘图失败:", e)

print("全部完成。")
