import pandas as pd
import pulp

# Battery capacity: 2 MWh (2000 kWh)
CAPACITY_KWH = 2000.0      

# Maximum charging/discharging power: 1 MW (1000 kW)
MAX_POWER_KW = 1000.0      

# Charging and discharging efficiency: 93.8%
ETA_CH = 0.938             
ETA_DIS = 0.938            

# 初始荷电状态: 容量的 50% [cite: 91]
INITIAL_SOC_KWH = 0.5 * CAPACITY_KWH  

DELTA_T = 1.0              


file_path = 'caseB_grid_battery_market_hourly.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: 找不到文件 {file_path}。请确保数据文件在同一目录下。")
    exit()

# 强制性要求: 必须执行并展示单位转换 [cite: 80]
# 理论依据: 电价单位为 GBP/MWh，功率单位为 kW。为了能量结算统一，将电价转换为 GBP/kWh。
# 换算公式: 1 MWh = 1000 kWh -> GBP/kWh = GBP/MWh / 1000
df['price_gbp_per_kwh'] = df['day_ahead_price_gbp_per_mwh'] / 1000.0

time_steps = range(len(df))


# Maximize Profit
model = pulp.LpProblem("Base_Battery_Arbitrage", pulp.LpMaximize)

# ==========================================
# Bound Variables
# 理论依据: 充放电功率在任何 delta_t=1h 内，均不可超过硬件逆变器的 1000 kW 极值。
p_ch = pulp.LpVariable.dicts("P_ch", time_steps, lowBound=0, upBound=MAX_POWER_KW, cat='Continuous')
p_dis = pulp.LpVariable.dicts("P_dis", time_steps, lowBound=0, upBound=MAX_POWER_KW, cat='Continuous')

# 理论依据: SOC 变量定义了电池在任何时刻内部的绝对化学能，极值被死锁在 0 到 2000 kWh 之间。
soc = pulp.LpVariable.dicts("SOC", time_steps, lowBound=0, upBound=CAPACITY_KWH, cat='Continuous')


# Profit = Revenue - Cost

model += pulp.lpSum(
    (p_dis[t] * df.loc[t, 'price_gbp_per_kwh'] - p_ch[t] * df.loc[t, 'price_gbp_per_kwh']) * DELTA_T 
    for t in time_steps
), "Total_Arbitrage_Profit"

# ==========================================
# 6. 系统状态转移与物理约束
# ==========================================
for t in time_steps:
    if t == 0:
        # T=0 时刻的能量守恒: 基于初始 SOC [cite: 91]
        model += soc[t] == INITIAL_SOC_KWH + (p_ch[t] * ETA_CH - p_dis[t] / ETA_DIS) * DELTA_T
    else:
        # 后续时刻的能量守恒: SOC(t) = SOC(t-1) + 净充入能量
        model += soc[t] == soc[t-1] + (p_ch[t] * ETA_CH - p_dis[t] / ETA_DIS) * DELTA_T

# 期末约束: 强制要求 60 天结束时，电池电量不能低于初始电量 [cite: 83]
# 理论依据: 避免模型为了最后一点利润将电池彻底榨干，确保下一个调度周期的物理连续性。
model += soc[len(time_steps)-1] >= INITIAL_SOC_KWH, "End_SOC_Requirement"

# ==========================================
# 7. 求解与指标计算
# ==========================================
print("正在计算最优调度策略...")
model.solve()

if pulp.LpStatus[model.status] == 'Optimal':
    total_profit = pulp.value(model.objective)
    
    # 初始化统计指标
    total_energy_throughput_kwh = 0.0
    total_discharge_kwh = 0.0
    results = []
    
    for t in time_steps:
        ch_kw = p_ch[t].varValue
        dis_kw = p_dis[t].varValue
        soc_val = soc[t].varValue
        
        # 记录每小时吞吐量 [cite: 82]
        total_energy_throughput_kwh += (ch_kw + dis_kw) * DELTA_T
        total_discharge_kwh += dis_kw * DELTA_T
        
        results.append({
            'Time': df.loc[t, 'timestamp'],
            'Price (GBP/MWh)': df.loc[t, 'day_ahead_price_gbp_per_mwh'],
            'Charge (kW)': ch_kw,
            'Discharge (kW)': dis_kw,
            'SOC (kWh)': soc_val
        })
        
    df_results = pd.DataFrame(results)
    
    # 计算精确的理论等效全循环 (EFC)
    # 理论依据: 每累计放电达到额定容量 (2000 kWh)，计为一次完整的等效全循环。
    equivalent_full_cycles = total_discharge_kwh / CAPACITY_KWH
    
    print("\n" + "="*50)
    print("基础模型求解成功 (Optimal)")
    print("="*50)
    print(f"1. 总利润 (Total Profit):        £{total_profit:,.2f}")
    print(f"2. 总能量吞吐 (Throughput):      {total_energy_throughput_kwh:,.2f} kWh")
    print(f"3. 等效全循环次数 (EFC):         {equivalent_full_cycles:.2f} 次")
    print(f"4. 期末实际 SOC:                 {df_results.iloc[-1]['SOC (kWh)']:.2f} kWh")
    print("="*50)
    
else:
    print(f"模型求解异常，状态: {pulp.LpStatus[model.status]}")