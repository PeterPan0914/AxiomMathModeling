"""鲁棒性与灵敏度分析框架模块。

提供六大组件的代码模板和分析框架，用于全面提升数学建模论文的鲁棒性分析质量。
取代传统的 OAT（单因素扰动）方法，实现多维度、结构化的鲁棒性验证。

六大组件：
1. 参数灵敏度分析（升级版 OAT）：基于置信区间、灵敏度指数、交互效应
2. 结构灵敏度分析：多模型规格对比、算法族对比
3. 数据灵敏度分析：K 折交叉验证、Bootstrap 置信区间、留一子组分析
4. 场景分析：关键自由参数扫描、阈值识别
5. 特征重要性分析：排列重要性、标准化回归系数
6. 稳定性验证：多次独立运行、解方差报告
"""

from __future__ import annotations


# ============================================================
# 组件 1：参数灵敏度分析（升级版 OAT）
# ============================================================

PARAMETER_SENSITIVITY_CODE = '''
# ============================================================
# 参数灵敏度分析（升级版 OAT）
# 核心改进：使用模型拟合的实际置信区间，报告灵敏度指数，检测交互效应
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def parameter_sensitivity_analysis(model_func, base_params, param_names, param_cis,
                                    n_samples=100, output_metric='result'):
    """升级版参数灵敏度分析。

    Args:
        model_func: 模型函数，接受参数字典，返回标量结果。
        base_params: 基准参数字典 {name: value}。
        param_names: 要分析的参数名列表。
        param_cis: 参数的置信区间字典 {name: (lower, upper)}。
        n_samples: 每个参数的采样点数。
        output_metric: 结果指标名称。

    Returns:
        sensitivity_df: 灵敏度分析结果 DataFrame。
        interaction_results: 交互效应分析结果。
    """
    base_result = model_func(base_params)
    results = []

    # ---- 单因素灵敏度（使用实际置信区间） ----
    for pname in param_names:
        lo, hi = param_cis[pname]
        param_values = np.linspace(lo, hi, n_samples)
        param_results = []

        for val in param_values:
            test_params = base_params.copy()
            test_params[pname] = val
            try:
                r = model_func(test_params)
                param_results.append(r)
            except Exception:
                param_results.append(np.nan)

        # 计算灵敏度指数：(最大结果 - 最小结果) / 基准结果
        valid = [r for r in param_results if not np.isnan(r)]
        if valid and base_result != 0:
            sensitivity_index = (max(valid) - min(valid)) / abs(base_result)
        else:
            sensitivity_index = np.nan

        results.append({
            '参数': pname,
            '基准值': base_params[pname],
            'CI下界': lo,
            'CI上界': hi,
            '结果最小值': min(valid) if valid else np.nan,
            '结果最大值': max(valid) if valid else np.nan,
            '灵敏度指数': sensitivity_index,
            '影响方向': '正向' if valid and param_results[-1] > param_results[0] else '负向',
        })

    sensitivity_df = pd.DataFrame(results)
    sensitivity_df = sensitivity_df.sort_values('灵敏度指数', ascending=False)

    # ---- 交互效应分析（同时扰动 2 个参数） ----
    interaction_results = []
    for p1, p2 in combinations(param_names, 2):
        lo1, hi1 = param_cis[p1]
        lo2, hi2 = param_cis[p2]
        grid_results = []

        for v1 in np.linspace(lo1, hi1, 10):
            for v2 in np.linspace(lo2, hi2, 10):
                test_params = base_params.copy()
                test_params[p1] = v1
                test_params[p2] = v2
                try:
                    r = model_func(test_params)
                    grid_results.append((v1, v2, r))
                except Exception:
                    pass

        if grid_results:
            vals = [g[2] for g in grid_results]
            # 交互效应强度：联合扰动的方差 vs 单独扰动方差之和
            joint_var = np.var(vals)
            solo_var_1 = sensitivity_df[sensitivity_df['参数']==p1]['灵敏度指数'].values
            solo_var_2 = sensitivity_df[sensitivity_df['参数']==p2]['灵敏度指数'].values
            interaction_strength = joint_var  # 简化度量

            interaction_results.append({
                '参数对': f'{p1} × {p2}',
                '联合扰动方差': joint_var,
                '交互效应强度': '强' if joint_var > np.median([np.var(list(np.linspace(param_cis[n][0], param_cis[n][1], 100))) for n in param_names]) else '弱',
            })

    interaction_df = pd.DataFrame(interaction_results) if interaction_results else pd.DataFrame()

    return sensitivity_df, interaction_df


def plot_sensitivity_tornado(sensitivity_df, base_result, save_path='sensitivity_tornado.png'):
    """绘制龙卷风图。

    Args:
        sensitivity_df: 灵敏度分析结果 DataFrame。
        base_result: 基准结果值。
        save_path: 图片保存路径。
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    df = sensitivity_df.sort_values('灵敏度指数', ascending=True)
    y_pos = range(len(df))

    # 计算偏差百分比
    left = [(r['结果最小值'] - base_result) / abs(base_result) * 100 for _, r in df.iterrows()]
    right = [(r['结果最大值'] - base_result) / abs(base_result) * 100 for _, r in df.iterrows()]

    ax.barh(y_pos, right, left=0, color=COLORS['secondary'], alpha=0.8, label='上界扰动')
    ax.barh(y_pos, left, left=0, color=COLORS['primary'], alpha=0.8, label='下界扰动')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['参数'])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('结果偏差 (%)')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【龙卷风图已保存】{save_path}")


def plot_sensitivity_heatmap(param1_name, param1_values, param2_name, param2_values,
                              model_func, base_params, save_path='sensitivity_heatmap.png'):
    """绘制两参数交互灵敏度热力图。

    Args:
        param1_name: 第一个参数名。
        param1_values: 第一个参数的取值数组。
        param2_name: 第二个参数名。
        param2_values: 第二个参数的取值数组。
        model_func: 模型函数。
        base_params: 基准参数字典。
        save_path: 图片保存路径。
    """
    grid = np.zeros((len(param1_values), len(param2_values)))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            test_params = base_params.copy()
            test_params[param1_name] = v1
            test_params[param2_name] = v2
            try:
                grid[i, j] = model_func(test_params)
            except Exception:
                grid[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(grid, xticklabels=[f'{v:.2f}' for v in param2_values],
                yticklabels=[f'{v:.2f}' for v in param1_values],
                cmap='RdYlBu_r', ax=ax, annot=False)
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【交互灵敏度热力图已保存】{save_path}")
    print(f"【热力图数据特征】最小值: {np.nanmin(grid):.4f}, 最大值: {np.nanmax(grid):.4f}, "
          f"极差比: {(np.nanmax(grid)-np.nanmin(grid))/abs(np.nanmean(grid))*100:.1f}%")
'''


# ============================================================
# 组件 2：结构灵敏度分析
# ============================================================

STRUCTURAL_SENSITIVITY_CODE = '''
# ============================================================
# 结构灵敏度分析
# 核心要求：比较至少 2 种模型规格，报告最佳模型及原因
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

def structural_sensitivity_regression(X, y, model_specs, cv=5):
    """回归模型的结构灵敏度分析。

    Args:
        X: 特征矩阵。
        y: 目标变量。
        model_specs: 字典 {模型名: (model_object, feature_subset)}，
                     feature_subset 为 None 时使用全部特征。
        cv: 交叉验证折数。

    Returns:
        comparison_df: 模型对比结果 DataFrame。
    """
    results = []

    for name, (model, features) in model_specs.items():
        if features is not None:
            X_sub = X[features]
        else:
            X_sub = X

        # 交叉验证
        cv_scores = cross_val_score(model, X_sub, y, cv=cv, scoring='r2')
        rmse_scores = cross_val_score(model, X_sub, y, cv=cv,
                                       scoring='neg_root_mean_squared_error')

        model.fit(X_sub, y)
        y_pred = model.predict(X_sub)

        results.append({
            '模型规格': name,
            '特征数': X_sub.shape[1],
            'R² (均值)': cv_scores.mean(),
            'R² (标准差)': cv_scores.std(),
            'RMSE (均值)': -rmse_scores.mean(),
            'RMSE (标准差)': rmse_scores.std(),
            '训练R²': r2_score(y, y_pred),
        })

    comparison_df = pd.DataFrame(results).sort_values('R² (均值)', ascending=False)

    print("=" * 60)
    print("【结构灵敏度分析 - 回归模型对比】")
    print(comparison_df.to_string(index=False))

    best = comparison_df.iloc[0]
    print(f"\n最佳模型: {best['模型规格']}")
    print(f"  R² = {best['R² (均值)']:.4f} (±{best['R² (标准差)']:.4f})")
    print(f"  RMSE = {best['RMSE (均值)']:.4f} (±{best['RMSE (标准差)']:.4f})")

    # 检查是否过拟合
    if best['训练R²'] - best['R² (均值)'] > 0.1:
        print(f"  警告: 训练R²({best['训练R²']:.4f}) 远高于交叉验证R²({best['R² (均值)']:.4f})，可能存在过拟合")
    print("=" * 60)

    return comparison_df


def structural_sensitivity_classification(X, y, model_specs, cv=5):
    """分类模型的结构灵敏度分析。

    Args:
        X: 特征矩阵。
        y: 目标变量。
        model_specs: 字典 {模型名: model_object}。
        cv: 交叉验证折数。

    Returns:
        comparison_df: 模型对比结果 DataFrame。
    """
    results = []

    for name, model in model_specs.items():
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

        model.fit(X, y)
        y_pred = model.predict(X)

        results.append({
            '模型规格': name,
            '准确率 (均值)': acc_scores.mean(),
            '准确率 (标准差)': acc_scores.std(),
            'F1 (均值)': f1_scores.mean(),
            'F1 (标准差)': f1_scores.std(),
            '训练准确率': accuracy_score(y, y_pred),
        })

    comparison_df = pd.DataFrame(results).sort_values('准确率 (均值)', ascending=False)

    print("=" * 60)
    print("【结构灵敏度分析 - 分类模型对比】")
    print(comparison_df.to_string(index=False))

    best = comparison_df.iloc[0]
    print(f"\n最佳模型: {best['模型规格']}")
    print(f"  准确率 = {best['准确率 (均值)']:.4f} (±{best['准确率 (标准差)']:.4f})")
    print(f"  F1 = {best['F1 (均值)']:.4f} (±{best['F1 (标准差)']:.4f})")
    print("=" * 60)

    return comparison_df


def plot_model_comparison(comparison_df, metric_col, save_path='model_comparison.png'):
    """绘制模型对比柱状图（含误差棒）。

    Args:
        comparison_df: 模型对比结果 DataFrame。
        metric_col: 要展示的指标列名（如 'R² (均值)'）。
        std_col: 对应的标准差列名。
        save_path: 图片保存路径。
    """
    std_col = metric_col.replace('(均值)', '(标准差)')

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(comparison_df))
    bars = ax.bar(x, comparison_df[metric_col],
                  yerr=comparison_df[std_col],
                  capsize=5, color=COLORS['primary'], alpha=0.8,
                  edgecolor='white', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['模型规格'], rotation=15, ha='right')
    ax.set_ylabel(metric_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注最佳值
    best_idx = comparison_df[metric_col].idxmax()
    ax.annotate('最佳', xy=(best_idx, comparison_df.loc[best_idx, metric_col]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, color=COLORS['secondary'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【模型对比图已保存】{save_path}")
'''


# ============================================================
# 组件 3：数据灵敏度分析
# ============================================================

DATA_SENSITIVITY_CODE = '''
# ============================================================
# 数据灵敏度分析
# 三大方法：K 折交叉验证、Bootstrap 置信区间、留一子组分析
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

def bootstrap_confidence_interval(model, X, y, n_bootstrap=1000, ci=0.95,
                                    metric_func=None, random_state=42):
    """Bootstrap 置信区间估计。

    Args:
        model: sklearn 兼容的模型对象。
        X: 特征矩阵。
        y: 目标变量。
        n_bootstrap: Bootstrap 采样次数。
        ci: 置信水平（默认 0.95）。
        metric_func: 评估函数，默认为 R²。
        random_state: 随机种子。

    Returns:
        ci_lower: 置信区间下界。
        ci_upper: 置信区间上界。
        bootstrap_scores: 所有 Bootstrap 分数。
    """
    if metric_func is None:
        from sklearn.metrics import r2_score
        metric_func = r2_score

    rng = np.random.RandomState(random_state)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # 有放回采样
        indices = rng.choice(len(X), size=len(X), replace=True)
        oob_indices = list(set(range(len(X))) - set(indices))

        X_boot, y_boot = X.iloc[indices], y.iloc[indices]

        if len(oob_indices) < 5:
            continue

        X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]

        try:
            model.fit(X_boot, y_boot)
            y_pred = model.predict(X_oob)
            score = metric_func(y_oob, y_pred)
            bootstrap_scores.append(score)
        except Exception:
            continue

    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_scores, alpha * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    print(f"【Bootstrap 置信区间 ({ci*100:.0f}% CI)】")
    print(f"   均值: {np.mean(bootstrap_scores):.4f}")
    print(f"   CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   标准差: {np.std(bootstrap_scores):.4f}")
    print(f"   有效采样数: {len(bootstrap_scores)}")

    return ci_lower, ci_upper, bootstrap_scores


def kfold_detailed_report(model, X, y, k=5):
    """K 折交叉验证详细报告。

    Args:
        model: sklearn 兼容的模型对象。
        X: 特征矩阵。
        y: 目标变量。
        k: 折数（默认 5）。

    Returns:
        fold_results: 每折结果 DataFrame。
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        fold_results.append({
            'Fold': fold_idx + 1,
            'R²': r2_score(y_val, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
            'MAE': mean_absolute_error(y_val, y_pred),
            '训练集大小': len(train_idx),
            '验证集大小': len(val_idx),
        })

    fold_df = pd.DataFrame(fold_results)

    print("=" * 60)
    print(f"【{k} 折交叉验证详细报告】")
    print(fold_df.to_string(index=False))
    print(f"\n汇总: R² = {fold_df['R²'].mean():.4f} (±{fold_df['R²'].std():.4f}), "
          f"RMSE = {fold_df['RMSE'].mean():.4f} (±{fold_df['RMSE'].std():.4f})")
    print("=" * 60)

    return fold_df


def leave_one_subgroup_out(X, y, group_column, model, metric_func=None):
    """留一子组分析。

    依次移除每个子组，评估模型在剩余数据上的表现，
    用于检测模型是否过度依赖某类样本。

    Args:
        X: 特征矩阵（需包含 group_column）。
        y: 目标变量。
        group_column: 分组列名。
        model: sklearn 兼容的模型对象。
        metric_func: 评估函数。

    Returns:
        subgroup_results: 子组分析结果 DataFrame。
    """
    if metric_func is None:
        from sklearn.metrics import r2_score
        metric_func = r2_score

    groups = X[group_column].unique()
    results = []

    # 全数据基准
    model.fit(X.drop(columns=[group_column]), y)
    y_pred_full = model.predict(X.drop(columns=[group_column]))
    full_score = metric_func(y, y_pred_full)

    for group in groups:
        # 留出该子组
        mask = X[group_column] != group
        X_sub = X[mask].drop(columns=[group_column])
        y_sub = y[mask]

        if len(X_sub) < 10:
            continue

        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_sub)
        score = metric_func(y_sub, y_pred)

        results.append({
            '移除子组': group,
            '子组样本数': (X[group_column] == group).sum(),
            '剩余样本数': len(X_sub),
            'R² (移除后)': score,
            'R² 变化': score - full_score,
            '影响程度': '显著' if abs(score - full_score) > 0.05 else '轻微',
        })

    subgroup_df = pd.DataFrame(results).sort_values('R² 变化', ascending=False)

    print("=" * 60)
    print("【留一子组分析】")
    print(f"全数据基准 R²: {full_score:.4f}")
    print(subgroup_df.to_string(index=False))

    sensitive_groups = subgroup_df[subgroup_df['影响程度'] == '显著']
    if len(sensitive_groups) > 0:
        print(f"\n敏感子组: {', '.join(sensitive_groups['移除子组'].astype(str))}")
        print("模型对这些子组的依赖度较高，需关注泛化能力。")
    else:
        print("\n模型对所有子组均表现稳定，泛化能力良好。")
    print("=" * 60)

    return subgroup_df


def plot_bootstrap_distribution(bootstrap_scores, ci_lower, ci_upper,
                                 metric_name='R²', save_path='bootstrap_ci.png'):
    """绘制 Bootstrap 分布图。

    Args:
        bootstrap_scores: Bootstrap 分数列表。
        ci_lower: 置信区间下界。
        ci_upper: 置信区间上界。
        metric_name: 指标名称。
        save_path: 图片保存路径。
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(bootstrap_scores, bins=40, color=COLORS['primary'], alpha=0.7,
            edgecolor='white', linewidth=0.5, density=True)

    ax.axvline(ci_lower, color=COLORS['secondary'], linestyle='--', linewidth=1.5,
               label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    ax.axvline(ci_upper, color=COLORS['secondary'], linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(bootstrap_scores), color=COLORS['tertiary'], linestyle='-',
               linewidth=1.5, label=f'均值: {np.mean(bootstrap_scores):.3f}')

    ax.set_xlabel(metric_name)
    ax.set_ylabel('密度')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【Bootstrap 分布图已保存】{save_path}")
'''


# ============================================================
# 组件 4：场景分析
# ============================================================

SCENARIO_ANALYSIS_CODE = '''
# ============================================================
# 场景分析
# 核心：识别关键自由参数，扫描 5+ 水平，发现临界阈值
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def scenario_sweep(model_func, free_param_name, free_param_levels,
                    base_params, output_names=None):
    """场景扫描分析。

    对关键自由参数（如准确率要求、风险容忍度、成本阈值等）
    在多个水平上扫描，观察最优策略如何随参数变化。

    Args:
        model_func: 模型函数，接受参数字典，返回结果字典或标量。
        free_param_name: 自由参数名称。
        free_param_levels: 自由参数的取值水平列表（至少 5 个）。
        base_params: 基准参数字典。
        output_names: 输出指标名称列表（当 model_func 返回字典时）。

    Returns:
        scenario_df: 场景分析结果 DataFrame。
        thresholds: 识别到的临界阈值列表。
    """
    results = []

    for level in free_param_levels:
        test_params = base_params.copy()
        test_params[free_param_name] = level

        try:
            output = model_func(test_params)

            if isinstance(output, dict):
                row = {free_param_name: level}
                row.update(output)
            else:
                row = {free_param_name: level, '结果': output}

            results.append(row)
        except Exception as e:
            results.append({free_param_name: level, '结果': np.nan, '错误': str(e)})

    scenario_df = pd.DataFrame(results)

    # ---- 识别临界阈值（结果发生跳跃的位置） ----
    thresholds = []
    if '结果' in scenario_df.columns:
        values = scenario_df['结果'].dropna().values
        if len(values) > 2:
            # 计算一阶差分的绝对值
            diffs = np.abs(np.diff(values))
            median_diff = np.median(diffs)

            # 超过中位数 3 倍的变化视为跳跃
            for i, d in enumerate(diffs):
                if d > median_diff * 3 and median_diff > 0:
                    threshold_val = scenario_df[free_param_name].iloc[i + 1]
                    thresholds.append({
                        '阈值位置': threshold_val,
                        '变化幅度': d,
                        '变化前': values[i],
                        '变化后': values[i + 1],
                    })

    print("=" * 60)
    print(f"【场景分析: {free_param_name} 扫描】")
    print(scenario_df.to_string(index=False))

    if thresholds:
        print(f"\n发现 {len(thresholds)} 个临界阈值:")
        for t in thresholds:
            print(f"  {free_param_name} = {t['阈值位置']}: "
                  f"结果从 {t['变化前']:.4f} 跳变到 {t['变化后']:.4f} (变化 {t['变化幅度']:.4f})")
    else:
        print("\n未发现显著临界阈值，结果随参数连续变化。")
    print("=" * 60)

    return scenario_df, thresholds


def plot_scenario_analysis(scenario_df, free_param_name, output_col='结果',
                            thresholds=None, save_path='scenario_analysis.png'):
    """绘制场景分析图。

    Args:
        scenario_df: 场景分析结果 DataFrame。
        free_param_name: 自由参数名称。
        output_col: 输出列名。
        thresholds: 临界阈值列表。
        save_path: 图片保存路径。
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    x = scenario_df[free_param_name]
    y = scenario_df[output_col]

    ax.plot(x, y, 'o-', color=COLORS['primary'], linewidth=2, markersize=6)

    # 标记临界阈值
    if thresholds:
        for t in thresholds:
            ax.axvline(t['阈值位置'], color=COLORS['secondary'], linestyle='--',
                       alpha=0.7, linewidth=1.5)
            ax.annotate(f"阈值: {t['阈值位置']}",
                        xy=(t['阈值位置'], y.mean()),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=9, color=COLORS['secondary'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))

    ax.set_xlabel(free_param_name)
    ax.set_ylabel(output_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【场景分析图已保存】{save_path}")


def multi_scenario_comparison(scenarios, model_func, base_params, save_path='multi_scenario.png'):
    """多场景对比分析。

    同时改变多个关键参数，比较不同场景下的结果。

    Args:
        scenarios: 字典 {场景名: {参数名: 参数值}}。
        model_func: 模型函数。
        base_params: 基准参数字典。
        save_path: 图片保存路径。
    """
    results = []

    for name, overrides in scenarios.items():
        params = base_params.copy()
        params.update(overrides)
        try:
            output = model_func(params)
            row = {'场景': name}
            if isinstance(output, dict):
                row.update(output)
            else:
                row['结果'] = output
            results.append(row)
        except Exception:
            results.append({'场景': name, '结果': np.nan})

    scenario_df = pd.DataFrame(results)

    print("=" * 60)
    print("【多场景对比分析】")
    print(scenario_df.to_string(index=False))
    print("=" * 60)

    return scenario_df
'''


# ============================================================
# 组件 5：特征重要性分析
# ============================================================

FEATURE_IMPORTANCE_CODE = '''
# ============================================================
# 特征重要性分析
# 排列重要性（ML 模型）和标准化回归系数（线性模型）
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def permutation_importance_analysis(model, X, y, feature_names=None,
                                      n_repeats=10, random_state=42):
    """排列重要性分析。

    Args:
        model: 已训练的 sklearn 模型。
        X: 特征 DataFrame。
        y: 目标变量。
        feature_names: 特征名称列表。
        n_repeats: 重复打乱次数。
        random_state: 随机种子。

    Returns:
        importance_df: 特征重要性 DataFrame。
    """
    from sklearn.inspection import permutation_importance

    if feature_names is None:
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'特征{i}' for i in range(X.shape[1])]

    result = permutation_importance(model, X, y, n_repeats=n_repeats,
                                     random_state=random_state, scoring='r2')

    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性均值': result.importances_mean,
        '重要性标准差': result.importances_std,
    }).sort_values('重要性均值', ascending=False)

    print("=" * 60)
    print("【排列重要性分析】")
    for i, row in importance_df.iterrows():
        print(f"  {row['特征']}: {row['重要性均值']:.4f} (±{row['重要性标准差']:.4f})")

    # 识别关键特征和次要特征
    threshold = importance_df['重要性均值'].quantile(0.75)
    key_features = importance_df[importance_df['重要性均值'] >= threshold]
    print(f"\n关键特征 (重要性 >= {threshold:.4f}): {', '.join(key_features['特征'].tolist())}")
    print("=" * 60)

    return importance_df


def standardized_coefficients(X, y, feature_names=None):
    """标准化回归系数分析。

    对数据进行 Z-score 标准化后拟合线性回归，
    系数的绝对值直接反映特征重要性。

    Args:
        X: 特征 DataFrame。
        y: 目标变量。
        feature_names: 特征名称列表。

    Returns:
        coef_df: 标准化系数 DataFrame。
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    if feature_names is None:
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'特征{i}' for i in range(X.shape[1])]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = (y - y.mean()) / y.std()

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    coef_df = pd.DataFrame({
        '特征': feature_names,
        '标准化系数': model.coef_,
        '绝对值': np.abs(model.coef_),
        '方向': ['正向' if c > 0 else '负向' for c in model.coef_],
    }).sort_values('绝对值', ascending=False)

    print("=" * 60)
    print("【标准化回归系数分析】")
    for _, row in coef_df.iterrows():
        bar = '█' * int(row['绝对值'] * 20)
        print(f"  {row['特征']}: {row['标准化系数']:+.4f} ({row['方向']}) {bar}")
    print("=" * 60)

    return coef_df


def plot_feature_importance(importance_df, save_path='feature_importance.png'):
    """绘制特征重要性图。

    Args:
        importance_df: 特征重要性 DataFrame（需含 '特征' 和 '重要性均值' 列）。
        save_path: 图片保存路径。
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    df = importance_df.sort_values('重要性均值', ascending=True)

    colors = [COLORS['tertiary'] if v >= df['重要性均值'].quantile(0.75)
              else COLORS['primary'] for v in df['重要性均值']]

    ax.barh(range(len(df)), df['重要性均值'], xerr=df.get('重要性标准差', None),
            color=colors, alpha=0.8, capsize=4, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['特征'])
    ax.set_xlabel('重要性')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【特征重要性图已保存】{save_path}")
'''


# ============================================================
# 组件 6：稳定性验证
# ============================================================

STABILITY_VERIFICATION_CODE = '''
# ============================================================
# 稳定性验证
# 多次独立运行优化算法，报告解方差
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def optimization_stability_test(optimizer_func, n_runs=10, random_seeds=None):
    """优化算法稳定性验证。

    对优化问题进行多次独立运行，评估解的稳定性。

    Args:
        optimizer_func: 优化函数，接受 random_state 参数，
                        返回 (最优值, 最优解, 其他信息)。
        n_runs: 运行次数。
        random_seeds: 随机种子列表。

    Returns:
        stability_df: 每次运行结果 DataFrame。
        summary: 稳定性汇总字典。
    """
    if random_seeds is None:
        random_seeds = list(range(n_runs))

    results = []
    for i, seed in enumerate(random_seeds[:n_runs]):
        try:
            best_value, best_solution, info = optimizer_func(random_state=seed)
            results.append({
                '运行序号': i + 1,
                '随机种子': seed,
                '最优值': best_value,
                '最优解': str(best_solution),
                '迭代次数': info.get('iterations', np.nan),
                '收敛时间': info.get('time', np.nan),
            })
        except Exception as e:
            results.append({
                '运行序号': i + 1,
                '随机种子': seed,
                '最优值': np.nan,
                '最优解': '失败',
                '错误': str(e),
            })

    stability_df = pd.DataFrame(results)
    valid_values = stability_df['最优值'].dropna()

    summary = {
        '运行次数': n_runs,
        '成功次数': len(valid_values),
        '最优值均值': valid_values.mean() if len(valid_values) > 0 else np.nan,
        '最优值标准差': valid_values.std() if len(valid_values) > 1 else np.nan,
        '变异系数(CV)': valid_values.std() / abs(valid_values.mean()) if len(valid_values) > 1 and valid_values.mean() != 0 else np.nan,
        '最优值范围': (valid_values.min(), valid_values.max()) if len(valid_values) > 0 else (np.nan, np.nan),
        '最佳运行': stability_df.loc[valid_values.idxmin(), '运行序号'] if len(valid_values) > 0 else np.nan,
    }

    # 稳定性评级
    cv = summary['变异系数(CV)']
    if not np.isnan(cv):
        if cv < 0.01:
            summary['稳定性评级'] = '非常稳定（CV < 1%）'
        elif cv < 0.05:
            summary['稳定性评级'] = '稳定（CV < 5%）'
        elif cv < 0.10:
            summary['稳定性评级'] = '较稳定（CV < 10%）'
        else:
            summary['稳定性评级'] = '不稳定（CV >= 10%），解具有非唯一性'

    print("=" * 60)
    print(f"【优化稳定性验证 ({n_runs} 次独立运行)】")
    print(stability_df.to_string(index=False))
    print(f"\n汇总:")
    print(f"  最优值: {summary['最优值均值']:.6f} (±{summary['最优值标准差']:.6f})")
    print(f"  变异系数: {summary['变异系数(CV)']*100:.2f}%")
    print(f"  稳定性评级: {summary['稳定性评级']}")

    if '非唯一' in summary['稳定性评级']:
        print("  建议: 解方差较大，应在论文中承认解的非唯一性，并报告多次运行的结果分布。")
    else:
        print("  结论: 优化算法收敛稳定，结果可靠。")
    print("=" * 60)

    return stability_df, summary


def plot_stability_boxplot(stability_df, save_path='stability_boxplot.png'):
    """绘制稳定性验证箱线图。

    Args:
        stability_df: 稳定性验证结果 DataFrame。
        save_path: 图片保存路径。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    valid_values = stability_df['最优值'].dropna()

    # 箱线图
    axes[0].boxplot(valid_values, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['light'], edgecolor=COLORS['primary']),
                     medianprops=dict(color=COLORS['secondary'], linewidth=2))
    axes[0].set_ylabel('最优值')
    axes[0].set_xticklabels(['所有运行'])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # 收敛轨迹
    axes[1].plot(stability_df['运行序号'], stability_df['最优值'],
                 'o-', color=COLORS['primary'], markersize=5)
    axes[1].axhline(valid_values.mean(), color=COLORS['secondary'], linestyle='--',
                     label=f'均值: {valid_values.mean():.4f}')
    axes[1].fill_between(stability_df['运行序号'],
                          valid_values.mean() - valid_values.std(),
                          valid_values.mean() + valid_values.std(),
                          alpha=0.2, color=COLORS['secondary'], label='±1σ')
    axes[1].set_xlabel('运行序号')
    axes[1].set_ylabel('最优值')
    axes[1].legend(frameon=False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"【稳定性验证图已保存】{save_path}")
'''


# ============================================================
# 完整框架提示词（供 CoderAgent 使用）
# ============================================================

ROBUSTNESS_CODER_PROMPT = """
---

# 鲁棒性与灵敏度分析框架（必须执行！）

完成主模型求解后，必须执行以下六大鲁棒性分析组件。每个组件都要生成代码、执行、输出关键数据特征，并生成可视化图表。

## 组件 1：参数灵敏度分析（升级版 OAT）

**不要使用任意的 ±10%、±20% 扰动范围！** 必须使用模型拟合产生的实际置信区间。

执行步骤：
1. 从模型拟合结果中提取参数的 95% 置信区间
2. 在置信区间范围内均匀采样，计算每个参数对输出指标的影响
3. 计算灵敏度指数 = (最大结果 - 最小结果) / |基准结果|
4. 对最重要的 2 个参数，进行联合扰动（交互效应分析）
5. 绘制龙卷风图和交互灵敏度热力图

输出要求：
```
print("【参数灵敏度分析结果】")
print(f"  最敏感参数: {param_name}, 灵敏度指数: {index:.4f}")
print(f"  参数变化范围内的结果波动: [{min_val:.4f}, {max_val:.4f}]")
print(f"  交互效应: {param1} × {param2} 的交互效应为 {强/弱}")
```

## 组件 2：结构灵敏度分析

必须比较至少 2 种不同的模型规格：

对于回归/预测问题：
- 至少测试 3 种特征组合（全特征、去除低重要性特征、仅核心特征）
- 或测试 2 种不同算法（如线性回归 vs 随机森林 vs XGBoost）
- 使用 5 折交叉验证比较 R² 和 RMSE

对于分类问题：
- 至少测试 2 种不同算法族（如 SVM vs 随机森林 vs Logistic 回归）
- 使用 5 折交叉验证比较准确率和 F1

对于优化问题：
- 至少使用 2 种不同的优化算法（如遗传算法 vs 模拟退火 vs 粒子群）
- 比较最优值和收敛速度

输出要求：
```
print("【结构灵敏度分析结果】")
print(f"  最佳模型: {model_name}, R²={r2:.4f}")
print(f"  次优模型: {model_name2}, R²={r2_2:.4f}")
print(f"  最佳模型的优势: R² 提升 {(r2-r2_2)/r2_2*100:.1f}%")
```

## 组件 3：数据灵敏度分析

三个子分析，全部执行：

### 3a. K 折交叉验证（K=5）
- 报告每折的 R²/RMSE/准确率
- 报告均值和标准差

### 3b. Bootstrap 置信区间（1000 次采样）
- 对核心指标（如 R²、预测误差）进行 Bootstrap 估计
- 报告 95% 置信区间
- 绘制 Bootstrap 分布图

### 3c. 留一子组分析
- 根据数据中的自然分组（如类别、区间），依次移除每个子组
- 观察模型性能变化
- 识别模型对哪些子组最敏感

输出要求：
```
print("【数据灵敏度分析结果】")
print(f"  5折CV: R²={mean:.4f} (±{std:.4f})")
print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  敏感子组: {subgroup_name} (移除后 R² 变化 {delta:+.4f})")
```

## 组件 4：场景分析

识别题目中的"自由参数"（如准确率要求、风险容忍度、成本约束阈值等），
在 5+ 个水平上扫描，观察最优策略如何变化。

执行步骤：
1. 识别关键自由参数（题目中的可变要求或约束阈值）
2. 设定 5-7 个水平（覆盖合理范围）
3. 对每个水平重新求解模型
4. 记录核心结果指标的变化
5. 识别临界阈值（结果发生跳跃的位置）
6. 绘制场景分析曲线（标记阈值）

输出要求：
```
print("【场景分析结果】")
print(f"  自由参数: {param_name}")
print(f"  在水平 {level1} 下，结果为 {result1}")
print(f"  在水平 {level2} 下，结果为 {result2}")
print(f"  临界阈值: {threshold} (结果在此处发生跳跃)")
```

## 组件 5：特征重要性分析

根据模型类型选择方法：

ML 模型（随机森林、XGBoost 等）：
- 使用排列重要性（permutation importance）
- 重复 10 次以获得稳定估计

线性模型（线性回归、Ridge 等）：
- 使用标准化回归系数
- 对特征进行 Z-score 标准化后比较系数绝对值

输出要求：
```
print("【特征重要性分析结果】")
print(f"  最重要特征: {feature1} (重要性={imp1:.4f})")
print(f"  次重要特征: {feature2} (重要性={imp2:.4f})")
print(f"  关键特征: {list of key features}")
```

## 组件 6：稳定性验证（优化类问题必须执行）

对于使用启发式算法（遗传算法、模拟退火等）的优化问题：

执行步骤：
1. 使用不同随机种子运行优化算法 10 次
2. 记录每次的最优值和最优解
3. 计算变异系数 CV = 标准差 / 均值
4. 评定稳定性等级

稳定性等级：
- CV < 1%: 非常稳定
- CV < 5%: 稳定
- CV < 10%: 较稳定
- CV >= 10%: 不稳定（需在论文中承认解的非唯一性）

输出要求：
```
print("【稳定性验证结果】")
print(f"  10次运行最优值: {mean:.6f} (±{std:.6f})")
print(f"  变异系数: {cv*100:.2f}%")
print(f"  稳定性评级: {rating}")
print(f"  最佳解出现于第 {best_run} 次运行")
```

## 总结输出（每个子任务完成后必须输出）

```python
print("=" * 60)
print("【鲁棒性分析汇总】")
print("1. 参数灵敏度: 最敏感参数为 {param}，灵敏度指数 {index:.4f}")
print("2. 结构灵敏度: 最佳模型 {model}，优于次优模型 {delta:.1f}%")
print("3. 数据灵敏度: 5折CV R²={r2:.4f}±{std:.4f}，Bootstrap CI=[{lo:.4f},{hi:.4f}]")
print("4. 场景分析: 自由参数 {param} 在 {threshold} 处存在临界阈值")
print("5. 特征重要性: 前3特征为 {f1}, {f2}, {f3}")
print("6. 稳定性: {rating}")
print("=" * 60)
```
"""


# ============================================================
# 写作提示词（供 WriterAgent 使用）
# ============================================================

ROBUSTNESS_WRITER_PROMPT = """
# 六、模型的分析与检验

## 6.1 灵敏度分析

本章从六个维度对模型的鲁棒性进行系统验证，包括参数灵敏度、结构灵敏度、数据灵敏度、场景分析、特征重要性和稳定性验证。

### 6.1.1 参数灵敏度分析

【写作要求】
- 说明选择了哪些关键参数及其选择依据
- 明确参数扰动范围来自模型拟合的 95% 置信区间（不是任意的 ±10%）
- 用表格展示各参数的灵敏度指数（从大到小排列）
- 分析参数间的交互效应
- 引用龙卷风图和交互灵敏度热力图
- 结论要具体：哪些参数最敏感，灵敏度指数是多少，对结果的实际影响有多大

### 6.1.2 结构灵敏度分析

【写作要求】
- 列出对比的所有模型规格（至少 2 种）
- 用表格展示各模型的交叉验证性能指标
- 说明最佳模型是哪个，为什么它更好（具体数值对比）
- 讨论不同模型规格之间的差异是否显著
- 如果差异不显著，说明模型选择对结论的稳健性

### 6.1.3 数据灵敏度分析

【写作要求】
- 报告 5 折交叉验证的详细结果（每折和汇总）
- 报告 Bootstrap 95% 置信区间
- 报告留一子组分析结果，指出模型对哪些子组最敏感
- 绘制 Bootstrap 分布图
- 评估模型的泛化能力和数据依赖性

### 6.1.4 场景分析

【写作要求】
- 识别题目中的关键自由参数（如准确率要求、风险容忍度等）
- 说明扫描了哪些水平（至少 5 个）
- 用表格和图表展示结果随参数变化的趋势
- 重点分析是否存在临界阈值（结果发生跳跃的位置）
- 讨论临界阈值的实际意义（对决策者意味着什么）

### 6.1.5 特征重要性分析

【写作要求】
- 列出所有特征的重要性排序
- 识别关键特征（重要性排名前 25%）
- 分析特征重要性的实际意义
- 引用特征重要性图

### 6.1.6 稳定性验证（优化类问题）

【写作要求】
- 报告多次独立运行的结果（均值、标准差、变异系数）
- 给出稳定性评级
- 如果不稳定（CV >= 10%），要诚实承认解的非唯一性
- 讨论结果的可靠性和可重复性

### 6.1.7 鲁棒性综合评估

【写作要求】
- 综合六个维度的分析结果，给出整体鲁棒性评价
- 指出模型的主要优势和潜在风险
- 对模型的适用范围给出明确限定
- 总字数 600-1000 字

【关键原则】
- 不要写"模型具有较好的鲁棒性"这种空话，要引用具体数据
- 不要回避问题：如果某维度分析发现模型不够稳健，要如实报告
- 每个结论都要有数据支撑（灵敏度指数、CV 值、置信区间等）
"""


# ============================================================
# 评审标准（供 ReviewerAgent 使用）
# ============================================================

ROBUSTNESS_REVIEW_CRITERIA = """
## 鲁棒性分析评审标准（额外 10 分，总分 110 分）

### 检查要点

#### 参数灵敏度（2 分）
- [ ] 是否使用了模型拟合的实际置信区间（而非任意 ±10%）？
- [ ] 是否计算了灵敏度指数（而非只说"结果变化不大"）？
- [ ] 是否分析了参数交互效应？
- [ ] 是否有龙卷风图或热力图？

#### 结构灵敏度（2 分）
- [ ] 是否比较了至少 2 种模型规格？
- [ ] 是否使用交叉验证进行对比？
- [ ] 是否说明了最佳模型及其优势？

#### 数据灵敏度（2 分）
- [ ] 是否进行了 K 折交叉验证？
- [ ] 是否进行了 Bootstrap 置信区间估计？
- [ ] 是否进行了留一子组分析？

#### 场景分析（2 分）
- [ ] 是否识别了关键自由参数？
- [ ] 是否扫描了 5+ 个水平？
- [ ] 是否识别了临界阈值？

#### 特征重要性（1 分）
- [ ] 是否报告了特征重要性排序？
- [ ] 使用的方法是否与模型类型匹配？

#### 稳定性验证（1 分，优化类问题必查）
- [ ] 是否进行了多次独立运行？
- [ ] 是否报告了变异系数和稳定性评级？

### 扣分标准
- 缺少参数灵敏度中的置信区间使用：-1 分
- 没有灵敏度指数只有定性描述：-1 分
- 结构灵敏度只比较了 1 种模型：-2 分
- 场景分析不足 5 个水平：-1 分
- 结论总是"模型稳健"但无数据支撑：-2 分
- 完全没有鲁棒性分析：-10 分
"""
