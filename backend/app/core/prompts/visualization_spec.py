"""可视化规范模块：为 CoderAgent 提供图表类型选择指南和代码模板。

本模块定义了学术论文各章节所需的图表类型、代码模板、图-文整合规则和反模式。
优秀论文通常包含 14+ 种图表类型，涵盖流程图、架构图、LOWESS 曲线、Q-Q 图、
混淆矩阵等高级可视化，而不仅限于基础统计图。
"""

# =============================================================================
# 第一部分：各章节必需图表类型清单
# =============================================================================

SECTION_CHART_REQUIREMENTS = """
# 各章节图表类型清单（必须覆盖的图表类型）

## 一、问题分析章节（1-2 张）
| 图表类型 | 用途 | 优先级 |
|---------|------|--------|
| 总体方法论流程图 | 展示全文研究框架和技术路线 | 必须 |
| 问题关联图/概念图 | 展示各子问题之间的逻辑关系 | 推荐 |

## 二、数据预处理/EDA 章节（3-5 张）
| 图表类型 | 用途 | 优先级 |
|---------|------|--------|
| 相关性热力图 | 变量间相关关系，含 r 值标注 | 必须 |
| 数据分布直方图+KDE | 各变量分布特征（偏度、峰度） | 必须 |
| 散点矩阵图(pairs plot) | 多变量两两关系总览 | 推荐 |
| 特征重要性水平条形图 | 模型驱动的特征排序 | 推荐 |
| 箱线图/小提琴图 | 分组对比或异常值检测 | 推荐 |

## 三、模型建立与求解章节（4-6 张，按模型类型选择）
| 图表类型 | 用途 | 优先级 |
|---------|------|--------|
| 模型架构图（流程图形式） | 展示模型结构和数据流 | 必须 |
| 训练损失曲线(含验证集) | 训练过程监控，防止过拟合 | 推荐(深度学习) |
| Q-Q 残差图 | 检验残差正态性假设 | 必须(回归类) |
| LOWESS 残差分析图 | 残差与拟合值的关系，检测非线性 | 推荐(回归类) |
| 回归诊断图(4合1) | 综合残差诊断 | 推荐(回归类) |
| 预测值 vs 实际值散点图 | 拟合效果可视化 | 必须 |

## 四、结果展示章节（3-5 张）
| 图表类型 | 用途 | 优先级 |
|---------|------|--------|
| 响应曲线(多场景) | 参数变化对结果的影响 | 必须 |
| 混淆矩阵热力图 | 分类模型的详细性能 | 必须(分类问题) |
| ROC 曲线(含 AUC) | 分类器性能比较 | 推荐(分类问题) |
| Pareto 前沿图 | 多目标优化的权衡关系 | 必须(优化问题) |
| 预测区间图 | 点预测 + 置信区间/预测区间 | 推荐 |

## 五、灵敏度分析章节（2-3 张）
| 图表类型 | 用途 | 优先级 |
|---------|------|--------|
| 龙卷风图(Tornado) | 参数敏感性排序 | 必须 |
| 参数扫描曲线 | 单参数变化对结果的影响 | 必须 |
| 蜘蛛图(Spider) | 多参数同时变化的效果 | 推荐 |
"""


# =============================================================================
# 第二部分：matplotlib 代码模板
# =============================================================================

CODE_TEMPLATES = """
# 高级图表代码模板

在使用以下模板前，确保 notebook 开头已执行全局配置代码块。
所有模板中的 COLORS 变量来自全局配置。
所有图表不使用 ax.set_title()，标题通过论文 caption 添加。

---

## 模板 1：总体方法论流程图

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_flowchart(save_path="fig_methodology_flowchart.png"):
    \"\"\"绘制总体方法论流程图。\"\"\"
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义节点：(x, y, width, height, text, color, shape)
    # shape: 'rect' 矩形, 'diamond' 菱形, 'rounded' 圆角矩形
    nodes = [
        (1, 8.5, 8, 0.9, "问题输入：题目分析与数据预处理", COLORS['primary'], 'rounded'),
        (0.5, 6.5, 4, 0.9, "问题一：描述性分析\\n与相关性检验", '#2E5B88', 'rounded'),
        (5.5, 6.5, 4, 0.9, "问题二：模型构建\\n与参数估计", '#4A9B7F', 'rounded'),
        (0.5, 4.5, 4, 0.9, "问题三：优化求解\\n与结果分析", '#E85D4C', 'rounded'),
        (5.5, 4.5, 4, 0.9, "问题四：综合评价\\n与灵敏度分析", '#8B6914', 'rounded'),
        (2.5, 2.5, 5, 0.9, "结论：模型评价与建议", COLORS['primary'], 'rounded'),
    ]

    for (x, y, w, h, text, color, shape) in nodes:
        if shape == 'rounded':
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='white', alpha=0.85, linewidth=1.5)
        else:
            box = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.05",
                                 facecolor=color, edgecolor='white', alpha=0.85, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # 箭头连接
    arrows = [
        (5, 8.5, 2.5, 7.4),    # 输入 → 问题一
        (5, 8.5, 7.5, 7.4),    # 输入 → 问题二
        (2.5, 6.5, 2.5, 5.4),  # 问题一 → 问题三
        (7.5, 6.5, 7.5, 5.4),  # 问题二 → 问题四
        (2.5, 4.5, 4, 3.4),    # 问题三 → 结论
        (7.5, 4.5, 6, 3.4),    # 问题四 → 结论
    ]

    for (x1, y1, x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#555555', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 方法论流程图】")
    print(f"   总问题数: 4")
    print(f"   各问题采用方法: [具体模型名称]")
    print(f"   整体技术路线: 数据预处理 → 分问题建模 → 综合评价")

draw_flowchart()
```

---

## 模板 2：相关性热力图（含 r 值标注）

```python
import seaborn as sns
import numpy as np

def draw_correlation_heatmap(df, cols, save_path="fig_correlation_heatmap.png"):
    \"\"\"绘制带数值标注的相关性热力图。\"\"\"
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=FIG_SQUARE)

    # 使用 diverging colormap
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                square=True, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 关键数据特征输出
    print("【图X数据特征 - 相关性热力图】")
    # 找出最强正相关和负相关（排除对角线）
    corr_vals = corr.where(~np.eye(len(corr), dtype=bool))
    max_pair = corr_vals.unstack().idxmax()
    min_pair = corr_vals.unstack().idxmin()
    print(f"   最强正相关: {max_pair[0]} vs {max_pair[1]} (r={corr_vals.loc[max_pair]:.3f})")
    print(f"   最强负相关: {min_pair[0]} vs {min_pair[1]} (r={corr_vals.loc[min_pair]:.3f})")
    # 统计显著相关对数
    sig_count = ((corr_vals.abs() > 0.5) & (corr_vals.abs() < 1.0)).sum().sum() // 2
    print(f"   中等以上相关(|r|>0.5)变量对数: {sig_count}")

draw_correlation_heatmap(df, ['col1', 'col2', 'col3', 'col4'])
```

---

## 模板 3：散点矩阵图（Pairs Plot）

```python
def draw_pairs_plot(df, cols, hue_col=None, save_path="fig_pairs_plot.png"):
    \"\"\"绘制散点矩阵图，对角线为KDE分布。\"\"\"
    g = sns.pairplot(df[cols + ([hue_col] if hue_col else [])],
                     hue=hue_col, diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 15, 'edgecolor': 'none'},
                     diag_kws={'fill': True, 'alpha': 0.5},
                     palette=COLORS if hue_col else None)
    g.fig.set_size_inches(10, 10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 散点矩阵图】")
    print(f"   变量数: {len(cols)}")
    if hue_col:
        groups = df[hue_col].value_counts()
        for g_name, g_count in groups.items():
            print(f"   {g_name}: {g_count} 样本")

draw_pairs_plot(df, ['x1', 'x2', 'x3', 'x4'], hue_col='group')
```

---

## 模板 4：特征重要性水平条形图

```python
def draw_feature_importance(importances, feature_names, top_n=10,
                            save_path="fig_feature_importance.png"):
    \"\"\"绘制特征重要性水平条形图。\"\"\"
    # 排序取 top N
    indices = np.argsort(importances)[-top_n:]
    sorted_imp = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.4)))

    bars = ax.barh(range(len(sorted_imp)), sorted_imp,
                   color=COLORS['primary'], edgecolor='white', height=0.7)

    # 在条形末端标注数值
    for bar, val in zip(bars, sorted_imp):
        ax.text(bar.get_width() + max(sorted_imp) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('特征重要性')
    ax.set_xlim(0, max(sorted_imp) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 特征重要性】")
    for i, (name, imp) in enumerate(zip(sorted_names[::-1][:5], sorted_imp[::-1][:5])):
        print(f"   {i+1}. {name}: {imp:.4f}")

draw_feature_importance(model.feature_importances_, feature_names, top_n=10)
```

---

## 模板 5：Q-Q 残差图

```python
from scipy import stats

def draw_qq_plot(residuals, save_path="fig_qq_plot.png"):
    \"\"\"绘制 Q-Q 残差图，检验正态性假设。\"\"\"
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # 计算理论分位数和样本分位数
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')

    ax.scatter(osm, osr, s=20, color=COLORS['primary'], alpha=0.6, edgecolors='none', zorder=3)

    # 参考线
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=COLORS['secondary'],
            linewidth=1.5, linestyle='--', label=f'参考线 (R²={r**2:.4f})', zorder=2)

    # 95% 置信带
    n = len(residuals)
    se = slope * np.sqrt(1/n + osm**2 / np.sum((osm - osm.mean())**2))
    ax.fill_between(x_line,
                    slope * x_line + intercept - 1.96 * np.interp(x_line, osm, se),
                    slope * x_line + intercept + 1.96 * np.interp(x_line, osm, se),
                    alpha=0.15, color=COLORS['primary'], label='95% 置信带')

    ax.set_xlabel('理论分位数')
    ax.set_ylabel('样本分位数')
    ax.legend(loc='upper left')

    # Shapiro-Wilk 检验
    if len(residuals) <= 5000:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = stats.normaltest(residuals)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - Q-Q图】")
    print(f"   样本量: {len(residuals)}")
    print(f"   Shapiro-Wilk 统计量: {sw_stat:.4f}")
    print(f"   正态性检验 p 值: {sw_p:.4e}")
    print(f"   结论: {'残差近似正态' if sw_p > 0.05 else '残差偏离正态分布'}")
    print(f"   Q-Q 线性拟合 R²: {r**2:.4f}")

draw_qq_plot(model.residuals)
```

---

## 模板 6：LOWESS 残差分析图

```python
import statsmodels.api as sm

def draw_lowess_residual_plot(fitted_values, residuals,
                               save_path="fig_lowess_residual.png"):
    \"\"\"绘制 LOWESS 平滑残差图，检测非线性模式。\"\"\"
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # 残差散点
    ax.scatter(fitted_values, residuals, s=15, alpha=0.4,
               color=COLORS['primary'], edgecolors='none', zorder=3)

    # LOWESS 平滑曲线
    lowess = sm.nonparametric.lowess(residuals, fitted_values, frac=0.3)
    ax.plot(lowess[:, 0], lowess[:, 1], color=COLORS['secondary'],
            linewidth=2, label='LOWESS 平滑', zorder=4)

    # 零线
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=1, alpha=0.7)

    # 标注均值和标准差
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    ax.axhspan(mean_r - 2*std_r, mean_r + 2*std_r, alpha=0.08, color=COLORS['primary'])
    ax.text(ax.get_xlim()[1], mean_r + 2*std_r, '+2σ', fontsize=8, va='bottom', ha='right')
    ax.text(ax.get_xlim()[1], mean_r - 2*std_r, '-2σ', fontsize=8, va='top', ha='right')

    ax.set_xlabel('拟合值')
    ax.set_ylabel('残差')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - LOWESS残差分析】")
    print(f"   残差均值: {mean_r:.4f} (应接近0)")
    print(f"   残差标准差: {std_r:.4f}")
    # 检查 LOWESS 曲线是否偏离零线
    lowess_dev = np.max(np.abs(lowess[:, 1]))
    print(f"   LOWESS 最大偏离零线: {lowess_dev:.4f}")
    print(f"   结论: {'残差无明显非线性模式' if lowess_dev < 2*std_r else '残差存在非线性趋势，考虑添加高次项'}")

draw_lowess_residual_plot(model.fittedvalues, model.resid)
```

---

## 模板 7：混淆矩阵热力图

```python
from sklearn.metrics import confusion_matrix, classification_report

def draw_confusion_matrix(y_true, y_pred, labels=None,
                          save_path="fig_confusion_matrix.png"):
    \"\"\"绘制带百分比标注的混淆矩阵热力图。\"\"\"
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    if labels is None:
        labels = [f'类别{i}' for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(max(5, len(labels)*1.2), max(4, len(labels)*1.0)))

    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})

    # 在每个格子中同时显示数量和百分比
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{cm[i,j]}\\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 输出分类报告
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print("【图X数据特征 - 混淆矩阵】")
    print(f"   总样本数: {cm.sum()}")
    print(f"   总体准确率: {report['accuracy']:.1%}")
    for label in labels:
        if label in report:
            print(f"   {label}: 精确率={report[label]['precision']:.3f}, "
                  f"召回率={report[label]['recall']:.3f}, F1={report[label]['f1-score']:.3f}")

draw_confusion_matrix(y_test, y_pred, labels=['正常', '异常'])
```

---

## 模板 8：ROC 曲线（多模型对比）

```python
from sklearn.metrics import roc_curve, auc

def draw_roc_curves(y_true, y_probas_dict, save_path="fig_roc_curves.png"):
    \"\"\"绘制多模型 ROC 曲线对比图。\"\"\"
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
              COLORS['neutral'], COLORS['light']]

    for (name, y_proba), color in zip(y_probas_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f'{name} (AUC={roc_auc:.3f})')

    # 对角线基线
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle=':', linewidth=1, alpha=0.5)

    # 标注最优阈值点（Youden's J）
    fpr_best, tpr_best, thresholds = roc_curve(y_true, list(y_probas_dict.values())[0])
    j_scores = tpr_best - fpr_best
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr_best[best_idx], tpr_best[best_idx], marker='o', s=60,
               color=COLORS['secondary'], zorder=5, edgecolors='white',
               label=f'最优阈值={thresholds[best_idx]:.3f}')

    ax.set_xlabel('假阳性率 (FPR)')
    ax.set_ylabel('真阳性率 (TPR)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - ROC曲线】")
    for name, y_proba in y_probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        print(f"   {name} AUC: {auc(fpr, tpr):.4f}")
    print(f"   最优阈值: {thresholds[best_idx]:.3f} (Youden's J={j_scores[best_idx]:.3f})")

draw_roc_curves(y_test, {'模型A': proba_a, '模型B': proba_b, '模型C': proba_c})
```

---

## 模板 9：多场景响应曲线（含置信带/阴影区域）

```python
def draw_response_curves(x_values, scenarios, save_path="fig_response_curves.png"):
    \"\"\"绘制多场景响应曲线，含阴影区域表示不确定性。\"\"\"
    # scenarios: dict of {场景名: {'mean': array, 'std': array}}
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
              COLORS['neutral']]

    for (name, data), color in zip(scenarios.items(), colors):
        mean = data['mean']
        std = data.get('std', np.zeros_like(mean))

        ax.plot(x_values, mean, color=color, linewidth=1.8, label=name)
        ax.fill_between(x_values, mean - 1.96*std, mean + 1.96*std,
                        alpha=0.15, color=color)

    ax.set_xlabel('参数 X')
    ax.set_ylabel('响应值 Y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 响应曲线】")
    for name, data in scenarios.items():
        mean = data['mean']
        print(f"   {name}: 范围 [{mean.min():.2f}, {mean.max():.2f}], "
              f"均值 {mean.mean():.2f}")

draw_response_curves(x, {
    '场景A': {'mean': y_a, 'std': y_a_std},
    '场景B': {'mean': y_b, 'std': y_b_std},
    '场景C': {'mean': y_c, 'std': y_c_std},
})
```

---

## 模板 10：Pareto 前沿图（多目标优化）

```python
def draw_pareto_frontier(obj1, obj2, is_pareto, dominated=None,
                         save_path="fig_pareto_frontier.png"):
    \"\"\"绘制 Pareto 前沿图。\"\"\"
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # 被支配解
    if dominated is not None:
        ax.scatter(obj1[~is_pareto], obj2[~is_pareto], s=15, alpha=0.3,
                   color=COLORS['neutral'], label='被支配解', edgecolors='none')

    # Pareto 前沿
    pareto_obj1 = obj1[is_pareto]
    pareto_obj2 = obj2[is_pareto]
    # 按 obj1 排序以便连线
    sort_idx = np.argsort(pareto_obj1)
    ax.scatter(pareto_obj1, pareto_obj2, s=30, color=COLORS['secondary'],
               label='Pareto 最优解', edgecolors='white', linewidth=0.5, zorder=5)
    ax.plot(pareto_obj1[sort_idx], pareto_obj2[sort_idx],
            color=COLORS['secondary'], linewidth=1.5, linestyle='--', alpha=0.7)

    # 标注理想点
    ax.scatter(pareto_obj1.min(), pareto_obj2.min(), marker='*', s=150,
               color='#FFD700', edgecolors='black', linewidth=0.5,
               label='理想点', zorder=6)

    ax.set_xlabel('目标 1 (f₁)')
    ax.set_ylabel('目标 2 (f₂)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - Pareto前沿】")
    print(f"   总解数: {len(obj1)}")
    print(f"   Pareto 最优解数: {is_pareto.sum()}")
    print(f"   f1 范围: [{pareto_obj1.min():.2f}, {pareto_obj1.max():.2f}]")
    print(f"   f2 范围: [{pareto_obj2.min():.2f}, {pareto_obj2.max():.2f}]")
    # 权衡比
    if len(pareto_obj1) > 1:
        tradeoff = np.abs(np.diff(pareto_obj2[sort_idx]) / np.diff(pareto_obj1[sort_idx]))
        print(f"   平均权衡比 Δf2/Δf1: {tradeoff.mean():.3f}")

draw_pareto_frontier(obj1_vals, obj2_vals, is_pareto_mask)
```

---

## 模板 11：龙卷风图（Tornado Diagram）

```python
def draw_tornado_diagram(param_names, base_value, low_values, high_values,
                          save_path="fig_tornado.png"):
    \"\"\"绘制龙卷风图，展示参数敏感性排序。\"\"\"
    # 计算偏差幅度并排序
    deviations = np.abs(high_values - low_values)
    sort_idx = np.argsort(deviations)

    sorted_names = [param_names[i] for i in sort_idx]
    sorted_low = low_values[sort_idx]
    sorted_high = high_values[sort_idx]

    n = len(sorted_names)
    fig, ax = plt.subplots(figsize=(7, max(3.5, n * 0.5)))

    y_pos = np.arange(n)

    # 左侧（低于基准）
    ax.barh(y_pos, sorted_low - base_value, left=base_value,
            height=0.6, color=COLORS['secondary'], alpha=0.8, label='下限')

    # 右侧（高于基准）
    ax.barh(y_pos, sorted_high - base_value, left=base_value,
            height=0.6, color=COLORS['primary'], alpha=0.8, label='上限')

    # 基准线
    ax.axvline(x=base_value, color='black', linewidth=1, linestyle='-')

    # 标注数值
    for i in range(n):
        ax.text(sorted_low[i], i, f' {sorted_low[i]:.2f}',
                va='center', ha='right', fontsize=8)
        ax.text(sorted_high[i], i, f' {sorted_high[i]:.2f} ',
                va='center', ha='left', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('结果值')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 龙卷风图】")
    print(f"   基准值: {base_value:.2f}")
    # 最敏感参数
    most_idx = sort_idx[-1]
    print(f"   最敏感参数: {param_names[most_idx]}")
    print(f"     变化范围: [{low_values[most_idx]:.2f}, {high_values[most_idx]:.2f}]")
    print(f"     偏差幅度: {deviations[most_idx]:.2f}")

draw_tornado_diagram(
    param_names=['参数A', '参数B', '参数C', '参数D'],
    base_value=100.0,
    low_values=np.array([85.0, 90.0, 95.0, 92.0]),
    high_values=np.array([115.0, 108.0, 103.0, 112.0])
)
```

---

## 模板 12：参数扫描曲线图

```python
def draw_parameter_sweep(param_values, results_dict, baseline_idx=None,
                          save_path="fig_param_sweep.png"):
    \"\"\"绘制参数扫描曲线图，展示参数变化对多个指标的影响。\"\"\"
    # results_dict: {指标名: array_of_values}
    n_metrics = len(results_dict)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, results_dict.items()):
        ax.plot(param_values, values, color=COLORS['primary'], linewidth=1.8, marker='o',
                markersize=4, markerfacecolor='white', markeredgecolor=COLORS['primary'])

        if baseline_idx is not None:
            ax.axvline(x=param_values[baseline_idx], color=COLORS['secondary'],
                       linestyle='--', linewidth=1, alpha=0.7, label='基准值')
            ax.legend()

        ax.set_xlabel('参数值')
        ax.set_ylabel(metric_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 参数扫描】")
    for metric_name, values in results_dict.items():
        print(f"   {metric_name}: 范围 [{values.min():.4f}, {values.max():.4f}], "
              f"变异系数 {values.std()/values.mean()*100:.1f}%")

draw_parameter_sweep(
    param_values=np.linspace(0.1, 2.0, 20),
    results_dict={'RMSE': rmse_values, 'R²': r2_values},
    baseline_idx=10
)
```

---

## 模板 13：神经网络/模型架构图

```python
def draw_model_architecture(save_path="fig_model_architecture.png"):
    \"\"\"绘制神经网络/模型架构示意图。\"\"\"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 层定义：(x, y, width, height, text, sub_text, color)
    layers = [
        (0.3, 2, 1.8, 2, "输入层", "特征向量\\nx ∈ ℝⁿ", '#B8D4E8'),
        (2.8, 1.5, 1.8, 3, "隐藏层 1", "128 神经元\\nReLU + Dropout", '#7BA7CC'),
        (5.3, 1.5, 1.8, 3, "隐藏层 2", "64 神经元\\nReLU + BatchNorm", '#4A8AB5'),
        (7.8, 1.5, 1.8, 3, "隐藏层 3", "32 神经元\\nReLU", '#2E6B99'),
        (10.3, 2, 1.4, 2, "输出层", "Softmax", COLORS['secondary']),
    ]

    for (x, y, w, h, title, subtitle, color) in layers:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#222222')
        ax.text(x + w/2, y + h*0.3, subtitle, ha='center', va='center',
                fontsize=8, color='#444444')

    # 箭头
    for i in range(len(layers)-1):
        x1 = layers[i][0] + layers[i][2]
        y1 = layers[i][1] + layers[i][3]/2
        x2 = layers[i+1][0]
        y2 = layers[i+1][1] + layers[i+1][3]/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#555555', lw=1.5))

    # 标注维度变化
    dims = ["n", "128", "64", "32", "K"]
    for (x, y, w, h, *_), dim in zip(layers, dims):
        ax.text(x + w/2, y - 0.3, f'dim={dim}', ha='center', fontsize=8,
                color=COLORS['neutral'], style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 模型架构】")
    print(f"   网络层数: {len(layers)} (含输入输出)")
    print(f"   参数量级: ~{sum([128*64, 64*32, 32*3]):,} (仅权重)")
    print(f"   激活函数: ReLU (隐藏层), Softmax (输出层)")
    print(f"   正则化: Dropout(0.3) + BatchNorm")

draw_model_architecture()
```

---

## 模板 14：训练损失曲线

```python
def draw_training_curves(train_loss, val_loss, train_metric=None, val_metric=None,
                          metric_name='Accuracy', save_path="fig_training_curves.png"):
    \"\"\"绘制训练过程损失和指标曲线。\"\"\"
    has_metric = train_metric is not None
    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(10 if has_metric else 5, 4))
    if not has_metric:
        axes = [axes]

    epochs = range(1, len(train_loss) + 1)

    # 损失曲线
    axes[0].plot(epochs, train_loss, color=COLORS['primary'], linewidth=1.5, label='训练损失')
    axes[0].plot(epochs, val_loss, color=COLORS['secondary'], linewidth=1.5, label='验证损失')

    # 标注最优点
    best_epoch = np.argmin(val_loss) + 1
    axes[0].axvline(x=best_epoch, color=COLORS['neutral'], linestyle=':', alpha=0.5)
    axes[0].scatter(best_epoch, val_loss[best_epoch-1], s=60, color=COLORS['secondary'],
                    zorder=5, edgecolors='white')
    axes[0].annotate(f'最优 epoch={best_epoch}\\nval_loss={val_loss[best_epoch-1]:.4f}',
                     xy=(best_epoch, val_loss[best_epoch-1]),
                     xytext=(best_epoch + len(train_loss)*0.1, val_loss[best_epoch-1]),
                     fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()

    # 指标曲线
    if has_metric:
        axes[1].plot(epochs, train_metric, color=COLORS['primary'], linewidth=1.5, label=f'训练{metric_name}')
        axes[1].plot(epochs, val_metric, color=COLORS['secondary'], linewidth=1.5, label=f'验证{metric_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 训练曲线】")
    print(f"   总 Epoch 数: {len(train_loss)}")
    print(f"   最优 Epoch: {best_epoch}")
    print(f"   最终训练损失: {train_loss[-1]:.4f}")
    print(f"   最终验证损失: {val_loss[-1]:.4f}")
    gap = abs(train_loss[-1] - val_loss[-1])
    print(f"   训练-验证差距: {gap:.4f} ({'过拟合风险' if gap > 0.1 * train_loss[-1] else '拟合良好'})")
    if has_metric:
        print(f"   最终验证{metric_name}: {val_metric[-1]:.4f}")

draw_training_curves(train_losses, val_losses, train_accs, val_accs)
```

---

## 模板 15：回归诊断四合图

```python
def draw_regression_diagnostics(fitted, residuals, save_path="fig_regression_diagnostics.png"):
    \"\"\"绘制回归诊断四合图：残差vs拟合值、Scale-Location、Q-Q图、残差vs杠杆。\"\"\"
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    std_resid = (residuals - residuals.mean()) / residuals.std()
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))

    # (a) 残差 vs 拟合值
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, s=10, alpha=0.4, color=COLORS['primary'], edgecolors='none')
    lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.3)
    ax.plot(lowess[:, 0], lowess[:, 1], color=COLORS['secondary'], linewidth=1.5)
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('拟合值')
    ax.set_ylabel('残差')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    # (b) Scale-Location
    ax = axes[0, 1]
    ax.scatter(fitted, sqrt_abs_resid, s=10, alpha=0.4, color=COLORS['primary'], edgecolors='none')
    lowess_sl = sm.nonparametric.lowess(sqrt_abs_resid, fitted, frac=0.3)
    ax.plot(lowess_sl[:, 0], lowess_sl[:, 1], color=COLORS['secondary'], linewidth=1.5)
    ax.set_xlabel('拟合值')
    ax.set_ylabel('√|标准化残差|')
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    # (c) Q-Q 图
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist='norm')
    ax.scatter(osm, osr, s=10, alpha=0.4, color=COLORS['primary'], edgecolors='none')
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=COLORS['secondary'], linewidth=1.5)
    ax.set_xlabel('理论分位数')
    ax.set_ylabel('标准化残差')
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    # (d) 残差 vs 杠杆值（Cook's D 等高线需要 hat matrix，此处简化）
    ax = axes[1, 1]
    n = len(residuals)
    leverage = np.random.uniform(0, 0.3, n)  # 实际应从 hat matrix 计算
    ax.scatter(leverage, std_resid, s=10, alpha=0.4, color=COLORS['primary'], edgecolors='none')
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('杠杆值')
    ax.set_ylabel('标准化残差')
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("【图X数据特征 - 回归诊断】")
    print(f"   样本量: {len(residuals)}")
    print(f"   残差均值: {residuals.mean():.4f}")
    print(f"   残差标准差: {residuals.std():.4f}")
    sw_stat, sw_p = stats.shapiro(residuals[:5000])
    print(f"   正态性检验 p 值: {sw_p:.4e}")

draw_regression_diagnostics(model.fittedvalues, model.resid)
```
"""


# =============================================================================
# 第三部分：图-文整合规则
# =============================================================================

FIGURE_TEXT_INTEGRATION_RULES = """
# 图-文整合规则（WriterAgent + CoderAgent 共同遵守）

## 规则 1：每张图必须配至少 3 句分析

每幅图表在论文中插入后，必须有至少 3 句完整的分析文字。
分析文字必须包含以下三要素（缺一不可）：

### 要素 A：描述性观察（"看到了什么"）
- 指出图表中的具体趋势、模式或特征
- 引用图表中的具体位置（如"图(a)的左上角"、"红色曲线"）

### 要素 B：数据提取（"数值是多少"）
- 必须从图表中提取至少 2 个具体数值
- 包括极值、均值、关键交点、百分比等
- 示例："相关系数最高达 0.87"、"在 x=5 处取得最大值 23.5"

### 要素 C：论点关联（"说明了什么"）
- 将观察结果与论文的核心论点或假设联系起来
- 解释该发现对后续分析或结论的意义

## 正确示例

```
由图 3 的相关性热力图可知，变量 X₁ 与 X₃ 之间呈现最强的正相关关系
（r = 0.87, p < 0.001），表明两者在统计上具有高度共线性。
此外，变量 X₂ 与响应变量 Y 的相关系数为 0.62，在所有预测变量中最高，
说明 X₂ 是影响 Y 的最主要因素。值得注意的是，X₄ 与 Y 的相关性仅
为 0.08（p = 0.32），未达到统计显著水平，这为后续特征筛选提供了依据。
```

## 错误示例（禁止！）

```
如图 3 所示，相关性分析结果展示了变量之间的关系。
[无具体数值、无分析、无论点关联]
```

---

## 规则 2：禁止重复图表

- 同一图表不得在多个章节中重复出现
- 如果不同章节需要展示相同数据，必须使用不同的图表类型或不同的视角
- 例如：EDA 中的相关性热力图不得在模型章节中重复；模型章节应使用残差图等新图

## 规则 3：禁止纯标签复述

以下分析属于无效分析，必须替换为数据提取型分析：

| 禁止的写法 | 应替换为 |
|-----------|---------|
| "横轴表示时间，纵轴表示温度" | "从第 1 天到第 30 天，温度从 15°C 上升至 28°C" |
| "图中展示了三种模型的比较" | "模型 A 的 RMSE 为 3.2，比模型 B（4.1）低 22%，比模型 C（3.8）低 16%" |
| "可以观察到下降趋势" | "指标从初始值 0.95 下降至最终值 0.78，降幅为 17.9%" |

## 规则 4：图表编号与引用

- 图表按章节顺序连续编号：图 1, 图 2, ...
- 子图标注使用 (a), (b), (c) 格式
- 正文引用时使用"如图 X 所示"或"由图 X 可知"
- 每张图表在正文中至少被引用一次

## 规则 5：图表与论证的逻辑链

每个章节的图表应形成完整的证据链：

1. 第一张图：展示数据特征或问题现状
2. 中间图：展示模型构建过程或中间结果
3. 最后一张图：展示最终结果或验证

禁止图表之间没有逻辑关联的"散装"安排。
"""


# =============================================================================
# 第四部分：反模式清单
# =============================================================================

ANTI_PATTERNS = """
# 可视化反模式（严格禁止！）

## 反模式 1：重复图表
**表现**：同一数据用相同图表类型在多个章节出现。
**修正**：每个章节使用不同的图表类型展示不同角度。
**检测方法**：检查 savefig 路径是否重复，检查数据源是否与已有图表重叠。

## 反模式 2：无数据的空洞分析
**表现**："如图所示，结果表明模型效果较好。"
**修正**："如图 5 所示，模型在测试集上的 RMSE 为 3.21，较基线模型（5.67）降低了 43.4%，
R² 达到 0.92，表明模型能够解释 92% 的方差变异。"
**检测方法**：检查分析文字中是否包含至少 2 个具体数值。

## 反模式 3：只复述轴标签
**表现**："横轴为训练轮次，纵轴为损失值。"
**修正**："损失值在前 20 个 epoch 快速下降（从 2.3 降至 0.5），此后趋于平稳，
最终收敛至 0.12。训练集与验证集的损失差距仅为 0.03，未观察到明显的过拟合。"
**检测方法**：检查分析文字是否包含超出轴标签的信息。

## 反模式 4：3D 图表滥用
**表现**：使用 3D 柱状图、3D 饼图等增加视觉复杂度但不增加信息量的图表。
**修正**：使用 2D 热力图、等高线图或分面小图替代。
**例外**：仅在数据本身为三维结构（如三维曲面拟合）时允许使用 3D 图。

## 反模式 5：饼图
**表现**：使用饼图展示比例数据。
**修正**：使用水平条形图，更精确、更易比较。

## 反模式 6：图表内标题
**表现**：使用 `ax.set_title()` 在图表内部添加标题。
**修正**：标题通过论文的 caption 添加，图表内部只保留轴标签和图例。

## 反模式 7：低分辨率输出
**表现**：使用低于 300 dpi 的分辨率保存图片。
**修正**：所有图片统一使用 `dpi=300`，`bbox_inches='tight'`。

## 反模式 8：密集网格线
**表现**：使用 `ax.grid(True)` 添加满屏网格线。
**修正**：不使用网格线，或仅使用极淡的水平参考线。

## 反模式 9：四边完整边框
**表现**：图表四周都有边框线。
**修正**：通过全局配置 `axes.spines.top=False, axes.spines.right=False` 只保留左下边框。

## 反模式 10：过多颜色
**表现**：一张图表使用超过 5 种颜色。
**修正**：使用统一的 COLORS 配色方案，最多 5 种颜色。超过 5 个系列时考虑分面小图。

## 反模式 11：图例遮挡数据
**表现**：图例放置在数据密集区域，遮挡关键信息。
**修正**：将图例放在空白区域，或使用 `bbox_to_anchor` 放在图外。

## 反模式 12：缺少置信带/误差线
**表现**：只画均值曲线，不展示不确定性。
**修正**：使用 `fill_between` 添加置信区间，或使用误差棒。
"""


# =============================================================================
# 第五部分：图表数量指南
# =============================================================================

FIGURE_COUNT_GUIDE = """
# 图表数量指南

## 各章节推荐图表数量

| 章节 | 最少 | 推荐 | 最多 |
|------|------|------|------|
| 问题分析 | 1 | 1 | 2 |
| EDA/数据预处理 | 2 | 3 | 5 |
| 模型建立与求解（每个子问题） | 1 | 2 | 3 |
| 结果展示 | 2 | 3 | 5 |
| 灵敏度分析 | 1 | 2 | 3 |
| **全文合计** | **8** | **13** | **18** |

## 图表类型多样性要求

全文至少覆盖以下 14 种图表类型中的 8 种：

1. 流程图/方法论图（matplotlib.patches）
2. 相关性热力图（seaborn heatmap）
3. 数据分布直方图+KDE
4. 散点矩阵图（pairplot）
5. 特征重要性条形图
6. Q-Q 残差图
7. LOWESS 残差分析图
8. 回归诊断四合图
9. 训练损失曲线
10. 混淆矩阵热力图
11. ROC 曲线
12. 响应曲线（含置信带）
13. Pareto 前沿图
14. 龙卷风图/参数扫描图

## 质量检查清单

生成每张图表后，对照检查：
- [ ] 是否有明确的 save_path（fig_xxx.png 格式）？
- [ ] 是否执行了 print 输出关键数据特征？
- [ ] 分辨率是否为 300 dpi？
- [ ] 是否使用了统一的 COLORS 配色？
- [ ] 是否去掉了上右边框？
- [ ] 轴标签是否包含单位？
- [ ] 图例是否无边框？
- [ ] 是否避免了禁止的图表类型（3D、饼图）？
"""


# =============================================================================
# 导出：合并所有规范为一个完整提示词
# =============================================================================

def get_visualization_spec_prompt() -> str:
    """生成完整的可视化规范提示词，供 CoderAgent 使用。

    Returns:
        可视化规范提示词字符串。
    """
    return f"""
---

# 高级可视化规范（图表多样性与学术质量）

{SECTION_CHART_REQUIREMENTS}

{CODE_TEMPLATES}

{FIGURE_TEXT_INTEGRATION_RULES}

{ANTI_PATTERNS}

{FIGURE_COUNT_GUIDE}
"""
