import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载净车辆曲线法估算结果
net = pd.read_excel(r'D:\competition\code\result1_1.xlsx')


# 2. 加载官方数据
official = pd.read_csv(r'D:\competition\code\official_docks_202503.csv', encoding='utf-8-sig')


net = net.dropna(subset=['station_id'])
official = official.dropna(subset=['station_id'])

net['station_id'] = net['station_id'].astype(float).astype(int).astype(str)
official['station_id'] = official['station_id'].astype(float).astype(int).astype(str)

# 3. 合并
merge = pd.merge(net[['station_id','capacity']], official[['station_id','docks_total']], on='station_id', how='inner')
merge = merge.dropna(subset=['capacity','docks_total'])
merge = merge[merge['docks_total'] > 0]

# 4. 误差分析
merge['diff_net'] = merge['capacity'] - merge['docks_total']
merge['ape_net'] = (merge['diff_net'].abs() / merge['docks_total']) * 100
print('净车辆曲线法 vs 官方 — MAPE:', merge['ape_net'].mean())

# 5. 校准
X = merge['capacity']
y = merge['docks_total']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
alpha = model.params['capacity']
beta = model.params['const']
r_squared = model.rsquared
print(f"校准模型：official ≈ {alpha:.3f} * net + {beta:.1f}")
print(f"R² = {r_squared:.3f}")

# 6. 添加预测列
merge['capacity_calib'] = alpha * merge['capacity'] + beta
merge['diff_calib'] = merge['capacity_calib'] - merge['docks_total']
merge['ape_calib'] = (merge['diff_calib'].abs() / merge['docks_total']) * 100
print('校准后 MAPE:', merge['ape_calib'].mean())

# 7. 保存结果
merge.to_csv(r'D:\competition\code\net_vs_official_calib.csv', index=False, encoding='utf-8-sig')

# 8. 可视化
plt.figure(figsize=(8,6))
plt.scatter(merge['docks_total'], merge['capacity'], label='净车辆曲线法', alpha=0.5)
plt.scatter(merge['docks_total'], merge['capacity_calib'], label='校准后', alpha=0.5)
plt.plot([merge['docks_total'].min(), merge['docks_total'].max()],
         [merge['docks_total'].min(), merge['docks_total'].max()],
         'k--', label='理想线')
plt.xlabel('官方容量')
plt.ylabel('估算容量')
plt.title('净车辆曲线法 vs 校准后估算对比')
plt.legend()
plt.grid(True)
plt.savefig(r'D:\competition\code\figures\净车辆曲线法对比图.png', dpi=300, bbox_inches='tight')
plt.close()
