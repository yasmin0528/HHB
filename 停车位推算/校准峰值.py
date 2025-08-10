
import pandas as pd


# 1. 加载估算数据（最大峰值法）
df_peak = pd.read_excel(r'D:\competition\code\各站点最大峰值.xlsx')
df_peak['station_id'] = df_peak['station_id'].astype(str)

# 2. 直接使用最大峰值数据
df_peak2 = df_peak[['station_id','capacity']].rename(columns={'capacity':'capacity_peak'})



# 3. 加载官方数据（直接用官方汇总结果）
df_off_sub = pd.read_csv(r'D:\competition\code\official_docks_202503.csv', encoding='utf-8-sig')


df_peak2 = df_peak2.dropna(subset=['station_id'])
df_off_sub = df_off_sub.dropna(subset=['station_id'])

df_peak2['station_id'] = df_peak2['station_id'].astype(float).astype(int).astype(str)
df_off_sub['station_id'] = df_off_sub['station_id'].astype(float).astype(int).astype(str)


# 4. 峰值法与官方对比（按station_id合并）
df_peak_calib = pd.merge(df_peak2, df_off_sub, on='station_id', how='inner')


print("合并后数量：", len(df_peak_calib))
df_peak_calib = df_peak_calib.dropna(subset=['capacity_peak','docks_total'])
print("去除缺失后数量：", len(df_peak_calib))
df_peak_calib = df_peak_calib[df_peak_calib['docks_total'] > 0]
print("去除0后数量：", len(df_peak_calib))
# 清洗
df_peak_calib = df_peak_calib.dropna(subset=['capacity_peak','docks_total'])
df_peak_calib = df_peak_calib[df_peak_calib['docks_total'] > 0]
print("合并后数量：", len(df_peak_calib))
df_peak_calib = df_peak_calib.dropna(subset=['capacity_peak','docks_total'])
print("去除缺失后数量：", len(df_peak_calib))
df_peak_calib = df_peak_calib[df_peak_calib['docks_total'] > 0]
print("去除0后数量：", len(df_peak_calib))


# 5. 计算误差指标
df_peak_calib['diff_peak'] = df_peak_calib['capacity_peak'] - df_peak_calib['docks_total']
df_peak_calib['ape_peak'] = (df_peak_calib['diff_peak'].abs() / df_peak_calib['docks_total']) * 100
print("峰值 vs 官方（NUM_DOCKS_AVAILABLE+NUM_DOCKS_DISABLED）— MAPE:", df_peak_calib['ape_peak'].mean())
df_peak_calib.to_csv(r'D:\competition\code\peak_vs_official.csv', index=False)



# 校准

import statsmodels.api as sm
import matplotlib.pyplot as plt
# 修正matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 构建线性模型（峰值法与官方数据）
X = df_peak_calib['capacity_peak']
y = df_peak_calib['docks_total']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

# 2. 输出回归结果
alpha = model.params['capacity_peak']
beta = model.params['const']
r_squared = model.rsquared
print(f"校准模型：official ≈ {alpha:.3f} * peak + {beta:.1f}")
print(f"R² = {r_squared:.3f}")

# 3. 添加预测列
df_peak_calib['capacity_calib'] = alpha * df_peak_calib['capacity_peak'] + beta
df_peak_calib['diff_calib'] = df_peak_calib['capacity_calib'] - df_peak_calib['docks_total']
df_peak_calib['ape_calib'] = (df_peak_calib['diff_calib'].abs() / df_peak_calib['docks_total']) * 100
mape_calib = df_peak_calib['ape_calib'].mean()
print(f"校准后 MAPE: {mape_calib:.2f}%")

# 4. 可视化对比图
plt.figure(figsize=(8,6))
plt.scatter(df_peak_calib['docks_total'], df_peak_calib['capacity_peak'], label='峰值法', alpha=0.5)
plt.scatter(df_peak_calib['docks_total'], df_peak_calib['capacity_calib'], label='校准后', alpha=0.5)
plt.plot([df_peak_calib['docks_total'].min(), df_peak_calib['docks_total'].max()],
         [df_peak_calib['docks_total'].min(), df_peak_calib['docks_total'].max()],
         'k--', label='理想线')
plt.xlabel('官方容量')
plt.ylabel('估算容量')
plt.title('峰值法 vs 校准后估算对比')
plt.legend()
plt.grid(True)
plt.savefig(r'd:\competition\code\figures\求停车位对比图.png', dpi=300, bbox_inches='tight')
plt.close()
