import pandas as pd
import os
import matplotlib.pyplot as plt



def plot_metric_vs_n(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    save_path: str = None,
):
    """
    在不同 n 下，画某个 metric 的曲线：
    - 横轴: n
    - 纵轴: metric_col
    - 不同 method / beta_star 画成不同曲线
    """
    if "n" not in df.columns:
        raise ValueError("DataFrame 中缺少列 'n'")
    if metric_col not in df.columns:
        raise ValueError(f"DataFrame 中缺少列 '{metric_col}'")

    plt.figure(figsize=(6, 4))

    # 按 method 和 beta_star 分组，分别画曲线
    group_cols = ["method"]
    if "beta_star" in df.columns:
        group_cols.append("beta_star")

    for keys, sub in df.groupby(group_cols):
        sub = sub.sort_values("n")

        # 组名 -> legend 标签
        if isinstance(keys, tuple):
            method = keys[0]
            beta = keys[1] if len(keys) > 1 else None
        else:
            method = keys
            beta = None

        if beta is not None:
            label = f"{method}, β*={beta}"
        else:
            label = f"{method}"

        plt.plot(
            sub["n"],
            sub[metric_col],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    plt.xlabel("n (sample size)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[PLOT] Saved figure to: {save_path}")

    plt.show()


def add_true_recovery_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 df 上增加一列 'TrueRecoveryRate':
      TrueRecoveryRate = DR / loop_time
    其中 DR 是 perfectly recovered 的次数，loop_time 是重复实验次数。
    """
    df = df.copy()
    if "DR" not in df.columns or "loop_time" not in df.columns:
        raise ValueError("DataFrame 中缺少 'DR' 或 'loop_time' 列")

    df["TrueRecoveryRate"] = df["DR"] / df["loop_time"].astype(float)
    return df

