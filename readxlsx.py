# analyze_bikeshare.py
# 功能：
# 1) 读取 Capital Bikeshare 的 CSV 数据（自动解析时间）
# 2) 构建借车/还车"事件流"，计算每站点的相对净车数曲线
# 3) 统计各站点：Max_Positive、Min_Negative、Required_Docks_Est = max(abs(Max_Positive), abs(Min_Negative))
# 4) 导出 station_capacity_estimates.csv
# 5) 生成论文图（PNG 300dpi + SVG）：Top-N 容量柱状图（带数值标签）、单站点曲线、时长分布、
#    按小时借车/还车需求（均带数值标签）
# 仅用 matplotlib（无 seaborn），每个图单独画布，适合论文直接插图

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 配置 ==========
INPUT_CSV = "202503-capitalbikeshare-tripdata.csv"  # 改成你的 CSV 文件名
OUTPUT_DIR = Path("figures")                         # 图表输出目录
TOP_N = 20                                           # Top-N 站点
DPI = 300                                            # 论文图分辨率
FONT_FAMILY = "DejaVu Sans"                          # 如需中文，请改为系统中的中文字体名，例如 "SimHei"

# ========== 基础设置 ==========
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR.mkdir(exist_ok=True)

def read_and_clean(csv_path: str) -> pd.DataFrame:
    """读取 CSV，解析时间，做基础清洗"""
    df = pd.read_csv(csv_path)

    # 列名检查（根据数据实际列名可调整）
    required_cols = ["started_at", "ended_at", "start_station_name", "end_station_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 请检查 CSV 列名。")

    # 自动解析时间；解析失败设为 NaT
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"]   = pd.to_datetime(df["ended_at"], errors="coerce")
    # 丢弃时间解析失败的行
    before = len(df)
    df = df.dropna(subset=["started_at", "ended_at"]).copy()
    dropped = before - len(df)
    print(f"🧹 丢弃无法解析时间的记录: {dropped} 条")

    # 如果ride_duration_min列不存在，则计算骑行时长
    if "ride_duration_min" not in df.columns:
        df["ride_duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
        print("📊 计算骑行时长...")
    else:
        print("📊 使用现有骑行时长数据...")

    return df

def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """构建借车/还车事件表并按时间排序；计算每站点累计净变化"""
    borrow_events = pd.DataFrame({
        "event_time": df["started_at"].values,
        "station": df["start_station_name"].values,
        "delta": -1
    })
    return_events = pd.DataFrame({
        "event_time": df["ended_at"].values,
        "station": df["end_station_name"].values,
        "delta": 1
    })
    events = pd.concat([borrow_events, return_events], ignore_index=True)
    events = events.dropna(subset=["event_time", "station"]).copy()
    events = events.sort_values("event_time", kind="mergesort").reset_index(drop=True)

    # 计算每站点的累计净变化（相对量）
    events["cum_change"] = events.groupby("station")["delta"].cumsum()
    return events

def estimate_capacity(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """统计每站点 max/min 及 Required_Docks_Est；合并 station_id（若存在）"""
    agg = events.groupby("station")["cum_change"].agg(["max", "min"]).rename(
        columns={"max": "Max_Positive", "min": "Min_Negative"}
    )
    
    # 修改容量计算逻辑：取最大正数和最小负数中绝对值大的
    agg["Required_Docks_Est"] = agg.apply(
        lambda row: max(abs(row["Max_Positive"]), abs(row["Min_Negative"])), 
        axis=1
    )

    # 合并 station_id（若 CSV 同时含 start_station_id / end_station_id）
    id_cols = []
    if "start_station_id" in df.columns:
        id_cols.append(("start_station_name", "start_station_id"))
    if "end_station_id" in df.columns:
        id_cols.append(("end_station_name", "end_station_id"))

    station_id_map = None
    if id_cols:
        pieces = []
        for name_col, id_col in id_cols:
            tmp = df[[name_col, id_col]].dropna().drop_duplicates()
            tmp = tmp.rename(columns={name_col: "station", id_col: "station_id"})
            pieces.append(tmp)
        if pieces:
            station_id_map = pd.concat(pieces, ignore_index=True).drop_duplicates("station")

    if station_id_map is not None:
        result = agg.reset_index().merge(station_id_map, on="station", how="left")
        result = result[["station_id", "station", "Max_Positive", "Min_Negative", "Required_Docks_Est"]]
    else:
        result = agg.reset_index()
        result.insert(0, "station_id", np.nan)

    result = result.sort_values("Required_Docks_Est", ascending=False).reset_index(drop=True)
    return result

# ========== 出图与导出 ==========
def save_fig(fig, filename_base: str):
    png_path = OUTPUT_DIR / f"{filename_base}.png"
    svg_path = OUTPUT_DIR / f"{filename_base}.svg"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 已导出图：{png_path}  |  {svg_path}")

def plot_topN_capacity_bars(res_df: pd.DataFrame, top_n: int = 20):
    """Top-N 站点容量估计柱状图（带数值标签）
    容量计算：max(abs(Max_Positive), abs(Min_Negative))
    """
    top = res_df.head(top_n)
    labels = top["station"].astype(str).tolist()
    values = top["Required_Docks_Est"].astype(float).values

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(values)), values)

    # 数值标签（整数显示）
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{int(round(val))}",
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Estimated Required Docks (max of abs values)")
    ax.set_title(f"Top-{top_n} Stations by Estimated Required Docks")
    save_fig(fig, f"top{top_n}_required_docks")

def plot_station_capacity_curve(events_df: pd.DataFrame, station_name: str):
    """单站点时间-相对净车数曲线"""
    sub = events_df.loc[events_df["station"] == station_name, ["event_time", "cum_change"]].copy()
    if sub.empty:
        print(f"⚠️ 站点无数据：{station_name}")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["event_time"].values, sub["cum_change"].values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Net Bikes (relative)")
    ax.set_title(f"Capacity Curve (Relative) - {station_name}")
    save_fig(fig, f"capacity_curve_{station_name.replace('/', '_')[:60]}")

def plot_duration_hist(df_: pd.DataFrame, bins: int = 60):
    """骑行时长分布直方图（计数）"""
    vals = df_["ride_duration_min"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins)
    ax.set_xlabel("Ride Duration (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ride Durations")
    save_fig(fig, "ride_duration_distribution")

def plot_hourly_demand(df_: pd.DataFrame):
    """按小时借车/还车需求柱状图（带数值标签）"""
    # 借车
    df_b = df_[["started_at"]].copy()
    df_b["hour"] = df_b["started_at"].dt.hour
    borrow_cnt = df_b.groupby("hour").size()

    # 还车
    df_r = df_[["ended_at"]].copy()
    df_r["hour"] = df_r["ended_at"].dt.hour
    return_cnt = df_r.groupby("hour").size()

    hours = np.arange(24)
    b_vals = borrow_cnt.reindex(hours, fill_value=0).values
    r_vals = return_cnt.reindex(hours, fill_value=0).values

    # 借车
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars_b = ax1.bar(hours, b_vals)
    for bar, val in zip(bars_b, b_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{int(val)}", ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(hours)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Borrow Count")
    ax1.set_title("Hourly Borrow Demand")
    save_fig(fig1, "hourly_borrow_demand")

    # 还车
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars_r = ax2.bar(hours, r_vals)
    for bar, val in zip(bars_r, r_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height(),
                 f"{int(val)}", ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(hours)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Return Count")
    ax2.set_title("Hourly Return Demand")
    save_fig(fig2, "hourly_return_demand")

def main():
    print("🚴 数据读取与清洗 ...")
    df = read_and_clean(INPUT_CSV)

    print("🧮 构建事件并计算累计净变化 ...")
    events = build_events(df)

    print("📊 估计各站点容量指标 ...")
    result = estimate_capacity(events, df)

    # 导出表格
    result_path = Path("station_capacity_estimates.csv")
    result.to_csv(result_path, index=False)
    print(f"✅ 容量统计已导出：{result_path}")

    # 生成论文图
    print("🖼️ 生成论文图 ...")
    plot_topN_capacity_bars(result, top_n=TOP_N)
    plot_duration_hist(df, bins=60)
    plot_hourly_demand(df)

    # Top-N 前 3 个站点，绘制时间曲线
    for s in result["station"].head(3).tolist():
        plot_station_capacity_curve(events, s)

    print("🎉 完成！图像在 ./figures/ ，表格在 station_capacity_estimates.csv")

if __name__ == "__main__":
    main()
