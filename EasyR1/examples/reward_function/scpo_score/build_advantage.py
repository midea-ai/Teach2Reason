from typing import List, Dict, Any
import math


def partition_by_base_reward(
        base_rewards: List[float],
        comp_scores: List[float],
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    对应 5.1：基于基础 reward 构造原始 advantage，并给出正/负/零样本划分。

    Args:
        base_rewards: 长度为 G 的基础 reward 列表
        comp_scores: 长度为 G 的 competition score 列表（这里仅用于长度校验，保持接口一致）
        tol: 判断正/负/零的浮点容忍度

    Returns:
        一个字典，包含：
        - group_size: G
        - base_mean: 基础 reward 的组均值
        - a_raw: 原始 centered reward / raw advantage
        - pos_idx: a_raw > 0 的索引
        - neg_idx: a_raw < 0 的索引
        - zero_idx: |a_raw| <= tol 的索引
        - is_degenerate: 是否是退化 group（即所有 a_raw 都近似为 0）
    """
    if len(base_rewards) != len(comp_scores):
        raise ValueError(
            f"Length mismatch: len(base_rewards)={len(base_rewards)} "
            f"!= len(comp_scores)={len(comp_scores)}"
        )
    if len(base_rewards) == 0:
        raise ValueError("Input lists must be non-empty.")

    G = len(base_rewards)
    base_mean = sum(base_rewards) / G
    a_raw = [r - base_mean for r in base_rewards]

    pos_idx = [i for i, a in enumerate(a_raw) if a > tol]
    neg_idx = [i for i, a in enumerate(a_raw) if a < -tol]
    zero_idx = [i for i, a in enumerate(a_raw) if abs(a) <= tol]

    is_degenerate = (len(pos_idx) == 0 and len(neg_idx) == 0)

    return {
        "group_size": G,
        "base_mean": base_mean,
        "a_raw": a_raw,
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "zero_idx": zero_idx,
        "is_degenerate": is_degenerate,
    }


def partition_degenerate_by_competition(
        base_rewards: List[float],
        comp_scores: List[float],
        gamma: float,
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    对应 5.2：当 group 退化（即所有 a_raw 近似为 0）时，
    基于 competition score 构造正/负样本划分。

    划分规则：
        P^-(tau) = {i | s_i <= tau}
        P^+(tau) = {i | s_i > tau}
    在所有候选 tau 中，选择满足
        sum_{i in P^-} s_i < gamma * sum_{i in P^+} s_i
    的“最大” tau。

    这里为了正确处理重复分数，tau 按 comp_scores 的唯一值遍历。

    Args:
        base_rewards: 长度为 G 的基础 reward 列表
        comp_scores: 长度为 G 的 competition score 列表
        gamma: 5.2 中的超参数 gamma
        tol: 判断退化 group 的浮点容忍度

    Returns:
        一个字典，包含：
        - applicable: 当前函数是否适用（即是否为退化 group）
        - success: 是否找到可行划分
        - a_raw: 原始 centered reward
        - pos_idx: 正样本索引
        - neg_idx: 负样本索引
        - threshold: 选中的 tau；若无可行划分则为 None
        - reason: 状态说明
    """
    if len(base_rewards) != len(comp_scores):
        raise ValueError(
            f"Length mismatch: len(base_rewards)={len(base_rewards)} "
            f"!= len(comp_scores)={len(comp_scores)}"
        )
    if len(base_rewards) == 0:
        raise ValueError("Input lists must be non-empty.")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")

    # 先检查是否真的是退化 group
    base_info = partition_by_base_reward(base_rewards, comp_scores, tol=tol)
    a_raw = base_info["a_raw"]
    is_degenerate = base_info["is_degenerate"]

    if not is_degenerate:
        return {
            "applicable": False,
            "success": False,
            "a_raw": a_raw,
            "pos_idx": [],
            "neg_idx": [],
            "threshold": None,
            "reason": "non_degenerate_group",
        }

    # 退化 group：按唯一 score 值从小到大遍历 tau
    unique_scores = sorted(set(comp_scores))

    best_tau = None
    best_neg_idx = []
    best_pos_idx = []

    for tau in unique_scores:
        neg_idx = [i for i, s in enumerate(comp_scores) if s <= tau]
        pos_idx = [i for i, s in enumerate(comp_scores) if s > tau]

        # 必须保证两侧都非空
        if len(neg_idx) == 0 or len(pos_idx) == 0:
            continue

        sum_neg = sum(comp_scores[i] for i in neg_idx)
        sum_pos = sum(comp_scores[i] for i in pos_idx)

        if sum_neg <= gamma * sum_pos:
            best_tau = tau
            best_neg_idx = neg_idx
            best_pos_idx = pos_idx

    if best_tau is None:
        return {
            "applicable": True,
            "success": False,
            "a_raw": a_raw,
            "pos_idx": [],
            "neg_idx": [],
            "threshold": None,
            "reason": "no_feasible_threshold",
        }

    return {
        "applicable": True,
        "success": True,
        "a_raw": a_raw,
        "pos_idx": best_pos_idx,
        "neg_idx": best_neg_idx,
        "threshold": best_tau,
        "reason": "ok",
    }


def _population_std(values: List[float]) -> float:
    """
    计算总体标准差（unbiased=False）。
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty.")
    mean_val = sum(values) / len(values)
    var = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def build_centered_competition_pattern(
        comp_scores: List[float],
) -> Dict[str, Any]:
    """
    对应新的 5.3 前半部分：
        p_i = s_i - mu
        mu = (1/G) * sum_j s_j

    Args:
        comp_scores: 长度为 G 的 competition score 列表

    Returns:
        一个字典，包含：
        - success: 是否成功
        - reason: 状态说明
        - group_size: G
        - mean: mu
        - p: 长度为 G 的全局中心化 pattern
        - p_sum: sum(p)，理论上应接近 0
    """
    if len(comp_scores) == 0:
        return {
            "success": False,
            "reason": "empty_comp_scores",
            "group_size": 0,
            "mean": None,
            "p": [],
            "p_sum": None,
        }

    G = len(comp_scores)
    mean_val = sum(comp_scores) / G
    p = [s - mean_val for s in comp_scores]
    p_sum = sum(p)

    return {
        "success": True,
        "reason": "ok",
        "group_size": G,
        "mean": mean_val,
        "p": p,
        "p_sum": p_sum,
    }


def reconstruct_base_advantage_with_lambda_centered(
        comp_scores: List[float],
        pos_idx: List[int],
        neg_idx: List[int],
        eps: float = 1e-8,
        std_tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    对应新的 5.3 后半部分（退化情形）：

    1) 先全局中心化：
         p_i = s_i - mu
    2) 计算
         lambda_min = max(
             -(G / |P^-|) * min_{i in P+} p_i,
              (G / |P+|) * max_{i in P^-} p_i
         )
    3) 取
         lambda = lambda_min + delta_lambda   (strict=True)
         lambda = lambda_min                  (strict=False)
    4) 构造
         hat_p_i = p_i + lambda,   i in P+
                 = p_i,            i in P-
    5) 得到
         a_base_i = (hat_p_i - mean(hat_p)) / (std(hat_p) + eps)

    Args:
        comp_scores: 长度为 G 的 competition score 列表
        pos_idx: 正样本索引
        neg_idx: 负样本索引
        eps: 数值稳定项
        std_tol: 判定 std 过小的阈值

    Returns:
        一个字典，包含：
        - success: 是否成功
        - reason: 状态说明
        - p: 全局中心化后的 pattern
        - p_mean: 原始 comp_scores 的均值 mu
        - lambda_min
        - lambda_val
        - hat_p
        - hat_p_mean
        - hat_p_std
        - a_base
    """
    G = len(comp_scores)

    if G == 0:
        return {
            "success": False,
            "reason": "empty_comp_scores",
            "p": [],
            "p_mean": None,
            "lambda_min": None,
            "lambda_val": None,
            "hat_p": [],
            "hat_p_mean": None,
            "hat_p_std": None,
            "a_base": [],
        }

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "success": False,
            "reason": "empty_pos_or_neg",
            "p": [],
            "p_mean": None,
            "lambda_min": None,
            "lambda_val": None,
            "hat_p": [0.0] * G,
            "hat_p_mean": None,
            "hat_p_std": None,
            "a_base": [0.0] * G,
        }

    if any(i < 0 or i >= G for i in pos_idx + neg_idx):
        raise ValueError("pos_idx or neg_idx contains out-of-range index.")
    if set(pos_idx).intersection(set(neg_idx)):
        raise ValueError("pos_idx and neg_idx must be disjoint.")

    # Step 1: 全局中心化
    pattern_info = build_centered_competition_pattern(comp_scores)
    if not pattern_info["success"]:
        return {
            "success": False,
            "reason": f"pattern_failed:{pattern_info['reason']}",
            "p": [],
            "p_mean": None,
            "lambda_min": None,
            "lambda_val": None,
            "hat_p": [0.0] * G,
            "hat_p_mean": None,
            "hat_p_std": None,
            "a_base": [0.0] * G,
        }

    p = pattern_info["p"]
    p_mean = pattern_info["mean"]

    # Step 2: 计算 lambda_min
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)

    min_pos_p = min(p[i] for i in pos_idx)
    max_neg_p = max(p[i] for i in neg_idx)

    lambda_min = max(
        -(G / num_neg) * min_pos_p,
        (G / num_pos) * max_neg_p,
    )

    lambda_val = max(lambda_min, 0.0)

    # Step 3: 构造 hat_p
    hat_p = [0.0] * G
    pos_set = set(pos_idx)
    neg_set = set(neg_idx)

    for i in range(G):
        if i in pos_set:
            hat_p[i] = p[i] + lambda_val
        elif i in neg_set:
            hat_p[i] = p[i]
        else:
            # 理论上在退化重构时，样本应已被完整划入 P+ 或 P-
            # 这里保守地保留原始 p_i
            hat_p[i] = p[i]

    # Step 4: 重新标准化得到 a_base
    hat_p_mean = sum(hat_p) / G
    hat_p_std = _population_std(hat_p)

    if hat_p_std < std_tol:
        return {
            "success": False,
            "reason": "hat_p_std_too_small",
            "p": p,
            "p_mean": p_mean,
            "lambda_min": lambda_min,
            "lambda_val": lambda_val,
            "hat_p": hat_p,
            "hat_p_mean": hat_p_mean,
            "hat_p_std": hat_p_std,
            "a_base": [0.0] * G,
        }

    a_base = [
        (x - hat_p_mean) / (hat_p_std + eps)
        for x in hat_p
    ]

    return {
        "success": True,
        "reason": "ok",
        "p": p,
        "p_mean": p_mean,
        "lambda_min": lambda_min,
        "lambda_val": lambda_val,
        "hat_p": hat_p,
        "hat_p_mean": hat_p_mean,
        "hat_p_std": hat_p_std,
        "a_base": a_base,
    }


def _population_std(values: List[float]) -> float:
    if len(values) == 0:
        raise ValueError("values must be non-empty.")
    mean_val = sum(values) / len(values)
    var = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def _standardize_subset(
        comp_scores: List[float],
        indices: List[int],
        eps: float = 1e-8,
        std_tol: float = 1e-12,
) -> Dict[str, Any]:
    if len(indices) == 0:
        return {
            "success": False,
            "reason": "empty_subset",
            "mean": None,
            "std": None,
            "pattern": {},
        }

    subset_scores = [comp_scores[i] for i in indices]
    mean_val = sum(subset_scores) / len(subset_scores)
    std_val = _population_std(subset_scores)

    if std_val < std_tol:
        return {
            "success": False,
            "reason": "std_too_small",
            "mean": mean_val,
            "std": std_val,
            "pattern": {},
        }

    pattern = {
        i: (comp_scores[i] - mean_val) / (std_val + eps)
        for i in indices
    }

    return {
        "success": True,
        "reason": "ok",
        "mean": mean_val,
        "std": std_val,
        "pattern": pattern,
    }


def build_side_standardized_patterns(
        comp_scores: List[float],
        pos_idx: List[int],
        neg_idx: List[int],
        eps: float = 1e-8,
        std_tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    分别在 P+ / P- 内标准化。
    即使一侧失败，另一侧仍然继续计算。
    """
    G = len(comp_scores)

    if any(i < 0 or i >= G for i in pos_idx + neg_idx):
        raise ValueError("pos_idx or neg_idx contains out-of-range index.")
    if set(pos_idx).intersection(set(neg_idx)):
        raise ValueError("pos_idx and neg_idx must be disjoint.")

    pos_info = _standardize_subset(comp_scores, pos_idx, eps=eps, std_tol=std_tol)
    neg_info = _standardize_subset(comp_scores, neg_idx, eps=eps, std_tol=std_tol)

    pattern_full = [0.0] * G
    if pos_info["success"]:
        for i, v in pos_info["pattern"].items():
            pattern_full[i] = v
    if neg_info["success"]:
        for i, v in neg_info["pattern"].items():
            pattern_full[i] = v

    return {
        "success": pos_info["success"] or neg_info["success"],  # 至少一侧成功即可
        "reason": "ok" if (pos_info["success"] or neg_info["success"]) else "both_sides_failed",
        "pos_success": pos_info["success"],
        "neg_success": neg_info["success"],
        "pos_reason": pos_info["reason"],
        "neg_reason": neg_info["reason"],
        "pos_pattern": pos_info["pattern"],
        "neg_pattern": neg_info["pattern"],
        "pos_mean": pos_info["mean"],
        "pos_std": pos_info["std"],
        "neg_mean": neg_info["mean"],
        "neg_std": neg_info["std"],
        "pattern_full": pattern_full,
    }


def compute_bonus_scaling_factors_from_side_patterns(
        a_base: List[float],
        pos_pattern: Dict[int, float],
        neg_pattern: Dict[int, float],
        pos_idx: List[int],
        neg_idx: List[int],
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.9,
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    基于分侧标准化 pattern 计算 eta。
    若某一侧 pattern 不可用（空 dict），则该侧 eta=0，另一侧照常计算。
    """
    G = len(a_base)

    if not (0.0 <= alpha_pos < 1.0):
        raise ValueError(f"alpha_pos must satisfy 0 <= alpha_pos < 1, got {alpha_pos}.")
    if not (0.0 <= alpha_neg < 1.0):
        raise ValueError(f"alpha_neg must satisfy 0 <= alpha_neg < 1, got {alpha_neg}.")
    if any(i < 0 or i >= G for i in pos_idx + neg_idx):
        raise ValueError("pos_idx or neg_idx contains out-of-range index.")
    if set(pos_idx).intersection(set(neg_idx)):
        raise ValueError("pos_idx and neg_idx must be disjoint.")

    # 正侧
    if len(pos_pattern) == 0:
        eta_pos_crit = None
        eta_pos = 0.0
        pos_constraint_idx = []
        pos_ratios = []
        pos_reason = "pos_pattern_unavailable"
    else:
        pos_constraint_idx = [i for i in pos_idx if i in pos_pattern and pos_pattern[i] < -tol]
        pos_ratios = [(i, a_base[i] / (-pos_pattern[i])) for i in pos_constraint_idx]

        if len(pos_ratios) == 0:
            eta_pos_crit = None
            eta_pos = 0.0
            pos_reason = "no_active_pos_constraint"
        else:
            eta_pos_crit = max(min(r for _, r in pos_ratios), 0.0)
            eta_pos = alpha_pos * eta_pos_crit
            pos_reason = "ok"

    # 负侧
    if len(neg_pattern) == 0:
        eta_neg_crit = None
        eta_neg = 0.0
        neg_constraint_idx = []
        neg_ratios = []
        neg_reason = "neg_pattern_unavailable"
    else:
        neg_constraint_idx = [i for i in neg_idx if i in neg_pattern and neg_pattern[i] > tol]
        neg_ratios = [(i, (-a_base[i]) / neg_pattern[i]) for i in neg_constraint_idx]

        if len(neg_ratios) == 0:
            eta_neg_crit = None
            eta_neg = 0.0
            neg_reason = "no_active_neg_constraint"
        else:
            eta_neg_crit = max(min(r for _, r in neg_ratios), 0.0)
            eta_neg = alpha_neg * eta_neg_crit
            neg_reason = "ok"

    return {
        "success": True,
        "reason": "ok",
        "eta_pos_crit": eta_pos_crit,
        "eta_neg_crit": eta_neg_crit,
        "eta_pos": eta_pos,
        "eta_neg": eta_neg,
        "pos_constraint_idx": pos_constraint_idx,
        "neg_constraint_idx": neg_constraint_idx,
        "pos_ratios": pos_ratios,
        "neg_ratios": neg_ratios,
        "pos_reason": pos_reason,
        "neg_reason": neg_reason,
    }


def build_reasoner_bonus_from_side_patterns(
        a_base: List[float],
        pos_pattern: Dict[int, float],
        neg_pattern: Dict[int, float],
        pos_idx: List[int],
        neg_idx: List[int],
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.9,
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    基于分侧标准化 pattern 构造 bonus。
    某一侧 pattern 不可用时，仅该侧 bonus=0，另一侧正常计算。
    """
    G = len(a_base)

    scaling_info = compute_bonus_scaling_factors_from_side_patterns(
        a_base=a_base,
        pos_pattern=pos_pattern,
        neg_pattern=neg_pattern,
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        tol=tol,
    )

    eta_pos = scaling_info["eta_pos"]
    eta_neg = scaling_info["eta_neg"]

    bonus = [0.0] * G
    pos_set = set(pos_idx)
    neg_set = set(neg_idx)

    for i in range(G):
        if i in pos_set:
            bonus[i] = eta_pos * pos_pattern[i] if i in pos_pattern else 0.0
        elif i in neg_set:
            bonus[i] = eta_neg * neg_pattern[i] if i in neg_pattern else 0.0
        else:
            bonus[i] = 0.0

    final_adv = [a_base[i] + bonus[i] for i in range(G)]

    # 只有 pattern 可用的一侧才检查约束
    constraint_ok_pos = all(final_adv[i] >= -tol for i in pos_idx if i in pos_pattern)
    constraint_ok_neg = all(final_adv[i] <= tol for i in neg_idx if i in neg_pattern)

    return {
        "success": True,
        "reason": "ok",
        "eta_pos_crit": scaling_info["eta_pos_crit"],
        "eta_neg_crit": scaling_info["eta_neg_crit"],
        "eta_pos": eta_pos,
        "eta_neg": eta_neg,
        "bonus": bonus,
        "final_adv": final_adv,
        "constraint_ok_pos": constraint_ok_pos,
        "constraint_ok_neg": constraint_ok_neg,
        "pos_constraint_idx": scaling_info["pos_constraint_idx"],
        "neg_constraint_idx": scaling_info["neg_constraint_idx"],
        "pos_ratios": scaling_info["pos_ratios"],
        "neg_ratios": scaling_info["neg_ratios"],
        "pos_reason": scaling_info["pos_reason"],
        "neg_reason": scaling_info["neg_reason"],
    }


def build_reasoner_bonus_from_comp_scores(
        comp_scores: List[float],
        a_base: List[float],
        pos_idx: List[int],
        neg_idx: List[int],
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.9,
        eps: float = 1e-8,
        std_tol: float = 1e-12,
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    一站式 bonus builder：
    一侧标准化失败时，仅该侧 bonus=0，另一侧照常计算。
    """
    pattern_info = build_side_standardized_patterns(
        comp_scores=comp_scores,
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        eps=eps,
        std_tol=std_tol,
    )

    # 两侧都失败，才整体失败
    if not pattern_info["success"]:
        return {
            "success": False,
            "reason": "both_sides_failed",
            "pos_success": False,
            "neg_success": False,
            "pos_pattern": {},
            "neg_pattern": {},
            "eta_pos_crit": None,
            "eta_neg_crit": None,
            "eta_pos": 0.0,
            "eta_neg": 0.0,
            "bonus": [0.0] * len(comp_scores),
            "final_adv": list(a_base),
            "constraint_ok_pos": False,
            "constraint_ok_neg": False,
        }

    bonus_info = build_reasoner_bonus_from_side_patterns(
        a_base=a_base,
        pos_pattern=pattern_info["pos_pattern"],
        neg_pattern=pattern_info["neg_pattern"],
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        tol=tol,
    )

    return {
        "success": True,
        "reason": "ok",
        "pos_success": pattern_info["pos_success"],
        "neg_success": pattern_info["neg_success"],
        "pos_reason": pattern_info["pos_reason"],
        "neg_reason": pattern_info["neg_reason"],
        "pos_pattern": pattern_info["pos_pattern"],
        "neg_pattern": pattern_info["neg_pattern"],
        "pos_mean": pattern_info["pos_mean"],
        "pos_std": pattern_info["pos_std"],
        "neg_mean": pattern_info["neg_mean"],
        "neg_std": pattern_info["neg_std"],
        "eta_pos_crit": bonus_info["eta_pos_crit"],
        "eta_neg_crit": bonus_info["eta_neg_crit"],
        "eta_pos": bonus_info["eta_pos"],
        "eta_neg": bonus_info["eta_neg"],
        "bonus": bonus_info["bonus"],
        "final_adv": bonus_info["final_adv"],
        "constraint_ok_pos": bonus_info["constraint_ok_pos"],
        "constraint_ok_neg": bonus_info["constraint_ok_neg"],
        "pos_constraint_idx": bonus_info["pos_constraint_idx"],
        "neg_constraint_idx": bonus_info["neg_constraint_idx"],
        "pos_ratios": bonus_info["pos_ratios"],
        "neg_ratios": bonus_info["neg_ratios"],
    }


def build_reasoner_training_signals_batch(
        base_rewards_batch: List[List[float]],
        comp_scores_batch: List[List[float]],
        gamma: float = 1.0,
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.9,
        eps: float = 1e-8,
        std_tol: float = 1e-12,
        tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    端到端构造 Reasoner 的训练信号（batch 级）。

    输入:
        base_rewards_batch: [B, G]
        comp_scores_batch: [B, G]

    输出:
        一个 dict，包含：
        - a_base_batch: [B, G]
        - bonus_batch: [B, G]
        - final_adv_batch: [B, G]
        - group_info: 长度为 B 的列表，每个元素都是该 group 的详细中间信息

    逻辑:
        1) 先用 5.1 基于 base reward 做初始划分
        2) 若非退化:
              a_base = a_raw
              再基于 comp_scores 构造 bonus
        3) 若退化:
              用 5.2 基于 comp_scores 构造划分
              再用 5.3 重构 a_base
              再用 4.3 / 5.4 构造 bonus
        4) 最终输出:
              final_adv = a_base + bonus

    注意:
        - 若退化 group 无可行划分，则该 group 的 a_base / bonus / final_adv 全为 0
        - 若 bonus 构造阶段两侧 pattern 都失败，则 bonus=0, final_adv=a_base
        - 若某一侧 pattern 失败，只有该侧 bonus=0，另一侧仍会正常计算
    """
    # -----------------------------
    # 基本输入检查
    # -----------------------------
    if len(base_rewards_batch) != len(comp_scores_batch):
        raise ValueError(
            f"Batch size mismatch: len(base_rewards_batch)={len(base_rewards_batch)} "
            f"!= len(comp_scores_batch)={len(comp_scores_batch)}"
        )
    if len(base_rewards_batch) == 0:
        raise ValueError("Input batch must be non-empty.")

    batch_size = len(base_rewards_batch)
    group_size = len(base_rewards_batch[0])

    for b in range(batch_size):
        if len(base_rewards_batch[b]) != group_size:
            raise ValueError("All rows in base_rewards_batch must have the same Group_size.")
        if len(comp_scores_batch[b]) != group_size:
            raise ValueError("All rows in comp_scores_batch must have the same Group_size.")

    # -----------------------------
    # 输出容器
    # -----------------------------
    a_base_batch: List[List[float]] = []
    bonus_batch: List[List[float]] = []
    final_adv_batch: List[List[float]] = []
    group_info: List[Dict[str, Any]] = []

    # -----------------------------
    # 逐 group 处理
    # -----------------------------
    for b in range(batch_size):
        base_rewards = base_rewards_batch[b]
        comp_scores = comp_scores_batch[b]
        G = len(base_rewards)

        # 先做 5.1 基础划分
        base_part = partition_by_base_reward(
            base_rewards=base_rewards,
            comp_scores=comp_scores,
            tol=tol,
        )

        info: Dict[str, Any] = {
            "group_index": b,
            "group_size": G,
            "base_rewards": list(base_rewards),
            "comp_scores": list(comp_scores),
            "is_degenerate": base_part["is_degenerate"],
            "a_raw": list(base_part["a_raw"]),
            "base_mean": base_part["base_mean"],
            "initial_pos_idx": list(base_part["pos_idx"]),
            "initial_neg_idx": list(base_part["neg_idx"]),
            "zero_idx": list(base_part["zero_idx"]),
        }

        # =========================================================
        # Case 1: 非退化 group
        # =========================================================
        if not base_part["is_degenerate"]:
            pos_idx = list(base_part["pos_idx"])
            neg_idx = list(base_part["neg_idx"])
            a_base = list(base_part["a_raw"])

            bonus_info = build_reasoner_bonus_from_comp_scores(
                comp_scores=comp_scores,
                a_base=a_base,
                pos_idx=pos_idx,
                neg_idx=neg_idx,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                eps=eps,
                std_tol=std_tol,
                tol=tol,
            )

            if bonus_info["success"]:
                bonus = list(bonus_info["bonus"])
                final_adv = list(bonus_info["final_adv"])
                status = "normal_bonus_ok"
            else:
                # bonus 不可用时，保守退化为 final_adv = a_base
                bonus = [0.0] * G
                final_adv = list(a_base)
                status = f"normal_bonus_failed:{bonus_info['reason']}"

            info.update({
                "status": status,
                "partition_source": "base_reward",
                "pos_idx": pos_idx,
                "neg_idx": neg_idx,
                "lambda_val": None,
                "a_base": list(a_base),
                "bonus": list(bonus),
                "final_adv": list(final_adv),
                "bonus_success": bonus_info["success"],
                "pos_success": bonus_info.get("pos_success", None),
                "neg_success": bonus_info.get("neg_success", None),
                "eta_pos_crit": bonus_info.get("eta_pos_crit", None),
                "eta_neg_crit": bonus_info.get("eta_neg_crit", None),
                "eta_pos": bonus_info.get("eta_pos", 0.0),
                "eta_neg": bonus_info.get("eta_neg", 0.0),
                "constraint_ok_pos": bonus_info.get("constraint_ok_pos", False),
                "constraint_ok_neg": bonus_info.get("constraint_ok_neg", False),
                "pos_pattern": bonus_info.get("pos_pattern", {}),
                "neg_pattern": bonus_info.get("neg_pattern", {}),
                "pos_reason": bonus_info.get("pos_reason", None),
                "neg_reason": bonus_info.get("neg_reason", None),
            })

        # =========================================================
        # Case 2: 退化 group
        # =========================================================
        else:
            deg_part = partition_degenerate_by_competition(
                base_rewards=base_rewards,
                comp_scores=comp_scores,
                gamma=gamma,
                tol=tol,
            )

            # 退化 group 无可行划分
            if not deg_part["success"]:
                a_base = [0.0] * G
                bonus = [0.0] * G
                final_adv = [0.0] * G

                info.update({
                    "status": f"degenerate_no_feasible_split:{deg_part['reason']}",
                    "partition_source": "competition_failed",
                    "pos_idx": [],
                    "neg_idx": [],
                    "threshold": deg_part.get("threshold", None),
                    "lambda_val": None,
                    "a_base": a_base,
                    "bonus": bonus,
                    "final_adv": final_adv,
                    "bonus_success": False,
                    "pos_success": False,
                    "neg_success": False,
                    "eta_pos_crit": None,
                    "eta_neg_crit": None,
                    "eta_pos": 0.0,
                    "eta_neg": 0.0,
                    "constraint_ok_pos": False,
                    "constraint_ok_neg": False,
                    "pos_pattern": {},
                    "neg_pattern": {},
                    "pos_reason": None,
                    "neg_reason": None,
                })

            else:
                pos_idx = list(deg_part["pos_idx"])
                neg_idx = list(deg_part["neg_idx"])

                recon_info = reconstruct_base_advantage_with_lambda_centered(
                    comp_scores=comp_scores,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    eps=eps,
                    std_tol=std_tol,
                )

                # 重构失败
                if not recon_info["success"]:
                    a_base = [0.0] * G
                    bonus = [0.0] * G
                    final_adv = [0.0] * G

                    info.update({
                        "status": f"degenerate_reconstruct_failed:{recon_info['reason']}",
                        "partition_source": "competition",
                        "pos_idx": pos_idx,
                        "neg_idx": neg_idx,
                        "threshold": deg_part.get("threshold", None),
                        "lambda_val": recon_info.get("lambda_val", None),
                        "lambda_min": recon_info.get("lambda_min", None),
                        "a_base": a_base,
                        "bonus": bonus,
                        "final_adv": final_adv,
                        "bonus_success": False,
                        "pos_success": False,
                        "neg_success": False,
                        "eta_pos_crit": None,
                        "eta_neg_crit": None,
                        "eta_pos": 0.0,
                        "eta_neg": 0.0,
                        "constraint_ok_pos": False,
                        "constraint_ok_neg": False,
                        "pos_pattern": {},
                        "neg_pattern": {},
                        "pos_reason": None,
                        "neg_reason": None,
                    })

                else:
                    a_base = list(recon_info["a_base"])

                    bonus_info = build_reasoner_bonus_from_comp_scores(
                        comp_scores=comp_scores,
                        a_base=a_base,
                        pos_idx=pos_idx,
                        neg_idx=neg_idx,
                        alpha_pos=alpha_pos,
                        alpha_neg=alpha_neg,
                        eps=eps,
                        std_tol=std_tol,
                        tol=tol,
                    )

                    if bonus_info["success"]:
                        bonus = list(bonus_info["bonus"])
                        final_adv = list(bonus_info["final_adv"])
                        status = "degenerate_bonus_ok"
                    else:
                        bonus = [0.0] * G
                        final_adv = list(a_base)
                        status = f"degenerate_bonus_failed:{bonus_info['reason']}"

                    info.update({
                        "status": status,
                        "partition_source": "competition",
                        "pos_idx": pos_idx,
                        "neg_idx": neg_idx,
                        "threshold": deg_part.get("threshold", None),
                        "lambda_val": recon_info.get("lambda_val", None),
                        "lambda_min": recon_info.get("lambda_min", None),
                        "p_centered": recon_info.get("p", []),
                        "hat_p": recon_info.get("hat_p", []),
                        "a_base": list(a_base),
                        "bonus": list(bonus),
                        "final_adv": list(final_adv),
                        "bonus_success": bonus_info["success"],
                        "pos_success": bonus_info.get("pos_success", None),
                        "neg_success": bonus_info.get("neg_success", None),
                        "eta_pos_crit": bonus_info.get("eta_pos_crit", None),
                        "eta_neg_crit": bonus_info.get("eta_neg_crit", None),
                        "eta_pos": bonus_info.get("eta_pos", 0.0),
                        "eta_neg": bonus_info.get("eta_neg", 0.0),
                        "constraint_ok_pos": bonus_info.get("constraint_ok_pos", False),
                        "constraint_ok_neg": bonus_info.get("constraint_ok_neg", False),
                        "pos_pattern": bonus_info.get("pos_pattern", {}),
                        "neg_pattern": bonus_info.get("neg_pattern", {}),
                        "pos_reason": bonus_info.get("pos_reason", None),
                        "neg_reason": bonus_info.get("neg_reason", None),
                    })

        a_base_batch.append(list(info["a_base"]))
        bonus_batch.append(list(info["bonus"]))
        final_adv_batch.append(list(info["final_adv"]))
        group_info.append(info)

    return {
        "batch_size": batch_size,
        "group_size": group_size,
        "a_base_batch": a_base_batch,
        "bonus_batch": bonus_batch,
        "final_adv_batch": final_adv_batch,
        "group_info": group_info,
    }
