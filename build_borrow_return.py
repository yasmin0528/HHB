# build_borrow_return.py  
import argparse
import sys
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Aggregate borrow/return counts per station.")
    parser.add_argument("--trips", required=False, help="Path to trip data CSV (default: 202503-capitalbikeshare-tripdata.csv if exists)")
    parser.add_argument("--out", default="borrow_return.csv", help="Output CSV path (default: borrow_return.csv)")
    parser.add_argument("--start-col", default="start_station_id", help="Column name for trip start station id")
    parser.add_argument("--end-col", default="end_station_id", help="Column name for trip end station id")
    args = parser.parse_args()

    # 默认文件名（同目录）
    default_trips = "202503-capitalbikeshare-tripdata.csv"
    trips_path = args.trips or (default_trips if os.path.exists(default_trips) else None)

    if trips_path is None:
        print("[ERROR] 没有提供 --trips，且当前目录也找不到默认文件 202503-capitalbikeshare-tripdata.csv")
        sys.exit(1)

    out_path = args.out
    start_col = args.start_col
    end_col = args.end_col

    try:
        preview = pd.read_csv(trips_path, nrows=0)
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {trips_path}\n{e}")
        sys.exit(1)

    cols = set(preview.columns)
    if start_col not in cols or end_col not in cols:
        print(f"[ERROR] CSV 缺少必要列：{start_col} 或 {end_col}\n已检测到的列：{sorted(cols)}")
        sys.exit(1)

    # 只读需要的两列，作为字符串读取
    usecols = [start_col, end_col]
    df = pd.read_csv(trips_path, usecols=usecols, dtype=str)

    # 借车统计（起点）
    borrow = (
        df[start_col]
        .value_counts(dropna=False)
        .rename_axis("station_id")
        .reset_index(name="borrow_count")
    )

    # 还车统计（终点）
    ret = (
        df[end_col]
        .value_counts(dropna=False)
        .rename_axis("station_id")
        .reset_index(name="return_count")
    )

    # 合并
    out = pd.merge(borrow, ret, on="station_id", how="outer").fillna(0)

    # 统一转成字符串（缺失值用空字符串）
    out["station_id"] = out["station_id"].astype(str).str.strip()
    out["borrow_count"] = out["borrow_count"].astype(int)
    out["return_count"] = out["return_count"].astype(int)

    # 去掉 station_id 为空的行
    out = out[out["station_id"] != ""]
    out = out.sort_values("station_id").reset_index(drop=True)

    # 导出
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] 写出 {out_path}")
    print(f"[INFO] 站点数: {len(out)}，借车总次数: {out['borrow_count'].sum()}，还车总次数: {out['return_count'].sum()}")
    print(f"[INFO] 来源文件: {trips_path}")

if __name__ == "__main__":
    main()
