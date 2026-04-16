# 详细实验方案

## 基于方向变化(DC)和LLM的智能投资策略研究

### 实验设计文档 v1.0

---

## 一、实验总览

### 1.1 实验目标

本实验方案旨在验证以下核心假设：

| 假设编号 | 假设内容 | 验证方式 |
|----------|----------|----------|
| H1 | 多源情感分析优于单一情感源 | 对比实验 |
| H2 | 政策事件检测能提升策略收益 | 消融实验 |
| H3 | 跨时区分析能捕捉市场领先-滞后关系 | 相关性分析 |
| H4 | 整合框架(MuSA-Enhanced)优于基线MuSA | 主实验 |

### 1.2 实验环境

```
实验硬件：
- CPU: Intel i9-13900K
- GPU: NVIDIA RTX 4090 (24GB)
- RAM: 64GB DDR5
- 存储: 2TB NVMe SSD

实验软件：
- Python 3.9+
- PyTorch 2.0+
- Stable-Baselines3
- TA-Lib
- CUDA 12.1
```

---

## 二、数据集清单

### 2.1 价格数据

| 数据集 | 股票数量 | 时间范围 | 频率 | 用途 |
|--------|----------|----------|------|------|
| SP500_10 | 10只 | 2018-2024 | 日线 | 主实验 |
| SP500_30 | 30只 | 2018-2024 | 日线 | 扩展实验 |
| DJIA_10 | 10只 | 2018-2024 | 日线 | 消融实验 |

### 2.2 情感数据

| 数据集 | 来源 | 时间范围 | 记录数 |
|--------|------|----------|--------|
| all_SentimentScore_SP500.csv | FinBERT+Benzinga | 2018-2024 | 350,416 |
| multi_SentimentScore_SP500.csv | 多源融合 | 2018-2024 | 700,832 |

### 2.3 政策事件数据（需构建）

| 字段 | 类型 | 说明 |
|------|------|------|
| event_id | string | 事件唯一标识 |
| date | datetime | 事件日期 |
| event_type | category | tariff/trade_war/conflict |
| title | string | 事件标题 |
| description | string | 详细描述 |
| intensity | float | 强度 [0,1] |
| affected_markets | list | 影响市场列表 |

**关键事件清单**（需收集）：

```
事件1: 2018-03-22 - Trump宣布对中国600亿美元商品加征关税
事件2: 2018-07-06 - 中美关税战第一轮生效
事件3: 2019-05-10 - 关税提高至25%
事件4: 2020-01-15 - 中美第一阶段贸易协议
事件5: 2022-02-24 - 俄乌战争爆发
事件6: 2022-03-07 - 全球能源危机（油价暴涨）
事件7: 2023-10-07 - 巴以冲突升级
事件8: 2024-03-01 - 钢铝关税扩大
```

---

## 三、实验1：LLM情感分析对比实验

### 3.1 实验目的

验证不同LLM在金融新闻情感分析上的效果差异，为多LLM融合提供依据。

### 3.2 实验设置

| 参数 | 值 |
|------|-----|
| 测试集大小 | 500条 |
| 模型 | FinBERT, GPT-4, DeepSeek |
| 评估指标 | Accuracy, Precision, Recall, F1 |

### 3.3 数据准备

```python
# 带标签新闻数据格式
{
    "news_id": str,
    "date": "YYYY-MM-DD",
    "headline": str,
    "sentiment_label": "positive/negative/neutral",  # 人工标注
    "sentiment_score": float  # [-1, 1]
}
```

### 3.4 实验步骤

```
Step 1: 收集500条金融新闻（涵盖关税、贸易战、冲突主题）
Step 2: 人工标注情感标签（3名标注者，取多数投票）
Step 3: 分别使用FinBERT、GPT-4、DeepSeek进行情感分析
Step 4: 计算各模型指标
Step 5: 统计分析显著性差异
```

### 3.5 预期结果

| 模型 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| FinBERT | 0.72±0.03 | 0.70 | 0.68 | 0.69 |
| GPT-4 | 0.78±0.02 | 0.76 | 0.75 | 0.75 |
| DeepSeek | 0.74±0.03 | 0.72 | 0.71 | 0.71 |
| 融合模型 | 0.82±0.02 | 0.80 | 0.79 | 0.79 |

### 3.6 输出文件

- `exp1_llm_comparison_results.csv` - 各模型指标
- `exp1_confusion_matrix.png` - 混淆矩阵可视化
- `exp1_statistical_significance.txt` - 统计显著性检验结果

---

## 四、实验2：政策事件影响分析

### 4.1 实验目的

量化政策事件对不同市场的冲击强度、持续时间和传导路径。

### 4.2 事件分类

| 事件类型 | 关键词 | 预期影响市场 |
|----------|--------|--------------|
| tariff | tariff, duty, import tax | 进出口相关板块 |
| trade_war | trade war, sanction, embargo | 全球化公司 |
| regional_conflict | war, invasion, military | 能源、国防板块 |

### 4.3 实验指标

```
1. 即时效应 (Immediate Effect)
   - 事件当天市场收益率
   
2. 延迟效应 (Delayed Effect)
   - 事件后1天、3天、5天、10天收益率
   
3. 传导效应 (Transmission Effect)
   - 跨市场相关系数
   - 领先-滞后关系
```

### 4.4 实验步骤

```
Step 1: 整理8个关键政策事件
Step 2: 计算事件前后各市场收益率
Step 3: 绘制事件窗口期收益曲线
Step 4: 分析跨市场传导路径
Step 5: 统计显著异常收益日期
```

### 4.5 可视化输出

- `exp2_event_impact_timeline.png` - 事件影响时间线
- `exp2_cross_market_heatmap.png` - 跨市场相关性热力图
- `exp2_lead_lag_analysis.png` - 领先-滞后分析图

---

## 五、实验3：跨时区套利策略回测

### 5.1 实验目的

验证跨时区套利的可行性和收益来源。

### 5.2 策略设计

#### 策略A：事件驱动型套利

```
策略逻辑：
1. 监测美国政策事件（关税、贸易战）
2. 观察美股反应方向（上涨/下跌）
3. 在亚洲期货/ETF市场开盘前建仓
4. 亚洲开盘后平仓获利

交易标的：
- Long: FXI (中国ETF), EWJ (日本ETF)
- Short: EWH (香港ETF)
- 对冲: VIX期货
```

#### 策略B：隔夜持仓套利

```
策略逻辑：
1. 美股收盘时确定当日趋势
2. 买入亚洲市场相关ETF
3. 亚洲开盘后平仓

持仓时间：
- 美股收盘 (16:00 EST) → 亚洲开盘 (09:30 local)
- 约8-16小时
```

### 5.3 回测参数

| 参数 | 值 |
|------|-----|
| 回测时间 | 2019-01-01 至 2024-12-31 |
| 初始资金 | $100,000 |
| 交易成本 | 0.1% |
| 滑点 | 0.05% |
| 最大持仓 | 20% |

### 5.4 评估指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| 年化收益率 | (1+总收益)^(252/交易天数)-1 | >15% |
| 夏普比率 | (收益率-无风险利率)/收益率标准差 | >1.0 |
| 最大回撤 | (峰值-谷值)/峰值 | <15% |
| 胜率 | 盈利交易数/总交易数 | >55% |
| 盈亏比 | 平均盈利/平均亏损 | >1.5 |

### 5.5 预期结果

| 策略 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |
|------|----------|----------|----------|------|
| 策略A | 12.5% | 0.95 | 12.3% | 58% |
| 策略B | 8.2% | 0.72 | 9.8% | 52% |
| 基准(持有) | 6.8% | 0.55 | 18.5% | - |

---

## 六、实验4：主实验 - 完整框架效果

### 6.1 实验设计

#### 实验组设置

| 组别 | 配置 | 说明 |
|------|------|------|
| G1 | CRP | 均匀随机组合（基线） |
| G2 | DC | 方向变化算法 |
| G3 | DC-SentimentMulti | DC + 多源情感 |
| G4 | MuSA-Enhanced (本文) | DC + 多LLM + 事件检测 + 时区感知 |

### 6.2 训练-测试划分

```
训练集: 2018-01-01 至 2021-12-31 (4年)
验证集: 2022-01-01 至 2022-12-31 (1年)
测试集: 2023-01-01 至 2024-12-31 (2年)
```

### 6.3 超参数设置

#### RL智能体 (TD3)

| 参数 | 值 |
|------|-----|
| learning_rate_actor | 3e-4 |
| learning_rate_critic | 3e-4 |
| buffer_size | 100000 |
| batch_size | 256 |
| gamma | 0.99 |
| tau | 0.005 |
| policy_noise | 0.2 |
| noise_clip | 0.5 |
| policy_freq | 2 |
| train_freq | 1 |
| gradient_steps | 1 |
| episode_length | 252 |

#### 市场观察器

| 模型 | 隐藏层维度 | 层数 | Dropout |
|------|------------|------|---------|
| MLP_1 | [128, 64] | 2 | 0.2 |
| LSTM_1 | 64 | 1 | 0.1 |
| DC_1 | 32 | 1 | - |
| Enhanced | 128 | 2 | 0.15 |

### 6.4 运行配置

```python
# config.py 设置
config.benchmark_algo = 'MASA-dc'  # 或 'MASA-SentimentMulti'
config.topK = 10
config.num_epochs = 500
config.enable_market_observer = True
config.enable_controller = True
config.pricePredModel = 'MA'
config.otherRef_indicator_ma_window = 5
config.dailyRetun_lookback = 20
config.risk_market = 0.015
config.cbf_gamma = 0.1
```

### 6.5 评估指标

#### 核心指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Annual Return | 年化收益率 | 几何平均 |
| Sharpe Ratio | 夏普比率 | (Return-Rf)/Std |
| Max Drawdown | 最大回撤 | 峰值-谷值 |
| Sortino Ratio | 索提诺比率 | (Return-Rf)/DownsideStd |
| Calmar Ratio | 卡玛比率 | Annual Return/MDD |
| Win Rate | 胜率 | 盈利天数/总天数 |
| Volatility | 波动率 | 日收益年化标准差 |

#### 风险指标

| 指标 | 说明 | 阈值 |
|------|------|------|
| VaR (95%) | 95%置信度风险价值 | <5% |
| CVaR (95%) | 条件风险价值 | <8% |
| skewness | 收益偏度 | >-0.5 |
| kurtosis | 收益峰度 | <5 |

### 6.6 统计检验

```
1. Wilcoxon符号秩检验：对比G3 vs G4收益差异
2. Diebold-Mariano检验：预测能力对比
3..bootstrap置信区间：年化收益95%CI
```

---

## 七、实验5：消融实验

### 7.1 实验设计

| 实验 | 去掉模块 | 预期收益下降 |
|------|----------|--------------|
| ABL-1 | 多LLM情感 | -2.5% |
| ABL-2 | 政策事件检测 | -3.0% |
| ABL-3 | 时区感知调整 | -1.5% |
| ABL-4 | CBF控制器 | +2.0% (高收益高风险) |
| ABL-5 | RL智能体 | -8.0% (接近DC) |

### 7.2 消融配置

```
配置1: 原始MuSA (DC-SentimentMulti) - 基线
配置2: + 多LLM情感分析
配置3: + 政策事件检测
配置4: + 时区感知调整
配置5: + CBF风险控制
配置6: 完整增强版 (ALL)
```

### 7.3 预期结果表格

| 配置 | 年化收益 | 夏普比率 | 最大回撤 | 波动率 |
|------|----------|----------|----------|--------|
| 基线 | 8.2% | 0.68 | 14.5% | 12.1% |
| +多LLM | 9.5% | 0.78 | 13.2% | 11.5% |
| +事件检测 | 10.1% | 0.82 | 12.8% | 11.2% |
| +时区感知 | 9.8% | 0.80 | 13.0% | 11.4% |
| +CBF控制 | 11.2% | 0.95 | 10.5% | 10.8% |
| 完整版 | 13.5% | 1.15 | 9.2% | 10.2% |

### 7.4 可视化

- `exp5_ablation_bar_chart.png` - 各模块收益贡献柱状图
- `exp5_risk_return_scatter.png` - 风险-收益散点图
- `exp5_training_curve.png` - 训练曲线对比

---

## 八、实验6：特殊市场环境测试

### 8.1 压力测试场景

| 场景 | 时间段 | 特征 |
|------|--------|------|
| 贸易战升级 | 2018-06 至 2019-12 | 高波动、政策密集 |
| COVID-19 | 2020-02 至 2020-04 | 极端下跌、流动性危机 |
| 俄乌冲突 | 2022-02 至 2022-04 | 能源价格暴涨 |
| 美联储加息 | 2022-03 至 2023-07 | 利率上行、股债双跌 |

### 8.2 测试指标

```
压力测试指标：
- 最大回撤是否在可接受范围
- 能否及时止损
- 资金曲线是否快速恢复
- 组合是否保持正收益
```

### 8.3 预期表现

| 场景 | 基线策略 | 增强策略 | 优势 |
|------|----------|----------|------|
| 贸易战 | -12.5% | -6.2% | +6.3% |
| COVID-19 | -28.3% | -15.8% | +12.5% |
| 俄乌冲突 | -18.5% | -9.8% | +8.7% |
| 加息周期 | -8.2% | -4.5% | +3.7% |

---

## 九、实验流程与时间安排

### 9.1 实验执行顺序

```
Week 1-2: 数据准备
├── 收集政策事件数据
├── 整理新闻数据集
└── 验证价格数据完整性

Week 3-4: 实验1 - LLM对比
├── 模型推理
├── 指标计算
└── 统计分析

Week 5-6: 实验2 - 事件影响
├── 事件窗口分析
├── 传导路径可视化
└── 报告撰写

Week 7-8: 实验3 - 套利回测
├── 策略实现
├── 回测执行
└── 参数优化

Week 9-12: 实验4 - 主实验
├── 模型训练
├── 超参数调优
└── 结果评估

Week 13-14: 实验5 - 消融
├── 消融配置运行
└── 对比分析

Week 15-16: 实验6 - 压力测试
├── 场景模拟
└── 鲁棒性验证
```

### 9.2 自动化脚本

```bash
# 运行全部实验
python run_all_experiments.py --mode full

# 运行单实验
python run_all_experiments.py --mode exp1_llm_compare
python run_all_experiments.py --mode exp4_main

# 生成报告
python run_all_experiments.py --mode report
```

---

## 十、结果输出格式

### 10.1 目录结构

```
res/
├── Experiment1_LLM_Compare/
│   ├── results/
│   │   ├── llm_metrics.csv
│   │   ├── confusion_matrices/
│   │   └── statistical_tests.txt
│   └── figures/
│
├── Experiment2_Event_Impact/
│   ├── results/
│   │   ├── event_returns.csv
│   │   └── transmission_analysis.csv
│   └── figures/
│
├── Experiment3_Arbitrage/
│   ├── results/
│   │   ├── strategy_A_metrics.csv
│   │   └── strategy_B_metrics.csv
│   └── figures/
│
├── Experiment4_Main/
│   ├── results/
│   │   ├── training_log.csv
│   │   ├── test_performance.csv
│   │   └── trades_log.csv
│   └── figures/
│       ├── portfolio_curve.png
│       ├── monthly_returns_heatmap.png
│       └── risk_metrics.png
│
├── Experiment5_Ablation/
│   ├── results/
│   │   └── ablation_summary.csv
│   └── figures/
│
└── Experiment6_StressTest/
    ├── results/
    │   └── stress_test_results.csv
    └── figures/
```

### 10.2 结果汇总表格式

```csv
Experiment,Config,Annual_Return,Sharpe_Ratio,Max_Drawdown,Volatility,Win_Rate,Sortino_Ratio
Exp4_Main,G1_CRP,4.2%,0.32,22.5%,15.8%,48%,0.28
Exp4_Main,G2_DC,6.8%,0.55,18.2%,13.2%,52%,0.48
Exp4_Main,G3_DC_Senti,8.2%,0.68,14.5%,12.1%,55%,0.62
Exp4_Main,G4_MuSA_Enhanced,13.5%,1.15,9.2%,10.2%,62%,1.05
```

---

## 十一、异常情况处理

### 11.1 数据问题

| 问题 | 处理方式 |
|------|----------|
| 缺失交易日 | 前向填充 + 标注 |
| 异常价格 | 标记为NaN，不参与计算 |
| 情感数据缺失 | 使用0（中性）填充 |

### 11.2 模型问题

| 问题 | 处理方式 |
|------|----------|
| 训练不收敛 | 降低学习率，检查数据标准化 |
| 过拟合 | 增加dropout，早停 |
| CUDA OOM | 减小batch_size |

### 11.3 回测问题

| 问题 | 处理方式 |
|------|----------|
| 交易滑点过大 | 调整滑点参数至0.1% |
| 收益为负 | 检查策略逻辑，记录实验 |

---

## 十二、附录

### 12.1 关键配置文件

```python
# config.py
class Config:
    # 数据配置
    market_name = 'SP500'
    topK = 10
    train_date_start = '2018-01-01'
    train_date_end = '2021-12-31'
    test_date_start = '2023-01-01'
    test_date_end = '2024-12-31'
    
    # 算法配置
    benchmark_algo = 'MASA-dc'
    enable_market_observer = True
    enable_controller = True
    
    # RL配置
    num_epochs = 500
    buffer_size = 100000
    
    # 风险配置
    risk_market = 0.015
    cbf_gamma = 0.1
```

### 12.2 评估指标代码

```python
def calculate_metrics(returns):
    """计算所有评估指标"""
    annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
    sharpe = annual_return / returns.std() / np.sqrt(252)
    cumulative = (1 + returns).cumprod()
    max_dd = (cumulative / cummax - 1).min()
    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': returns.std() * np.sqrt(252)
    }
```

---

**文档版本**: v1.0
**创建日期**: 2026-04-15
**最后更新**: 2026-04-15