import time
import numpy as np
import pandas as pd

import os
from Lasso_num_exp import Lasso_num_exp
from onlineFSA_numerical_exp import onlineFSA_numexp

from tqdm import tqdm
import itertools

# =========================
# 1) 通用汇总：从 result_mat 得到 DR/PCD/RMSE/Runtime
# =========================
def summarize_result_mat(result_mat: np.ndarray, k_true: int):
    """
    result_mat columns (assumed by your code):
      onlineFSA_numexp:
        [seed, eta, hit, rmse, time]
      Lasso_num_exp:
        [seed, lbd_sel, hit, rmse, time]
    So hit is col=2, rmse col=3, time col=4 in both.
    """
    hit = result_mat[:, 2]
    rmse = result_mat[:, 3]
    runtime = result_mat[:, 4]

    DR = int(np.sum(hit == k_true))          # perfect recovery count
    PCD = float(np.mean(hit / k_true))       # proportion correctly discovered
    rmse_mean = float(np.mean(rmse))
    runtime_mean = float(np.mean(runtime))

    return {
        "DR": DR,
        "PCD": PCD,
        "RMSE_mean": rmse_mean,
        "Runtime_mean": runtime_mean,
    }




# =========================
# 2) 两个“适配器”：把不同方法的调用方式统一成 (gen_data, loop_time, **kwargs)->result_mat
# =========================
def run_onlineFSA(gen_data, loop_time, eta=0.01, lbd=0):
    # onlineFSA_numexp(gen_data, eta, lbd, loop_time)
    return onlineFSA_numexp(gen_data, eta=eta, lbd=lbd, loop_time=loop_time)

def run_lasso(gen_data, loop_time, lbd_para):
    # Lasso_num_exp(gen_dat, lbd_para, loop_time)
    return Lasso_num_exp(gen_dat=gen_data, lbd_para=lbd_para, loop_time=loop_time)


# =========================
# 3) 通用 grid search 引擎
# =========================
def run_grid_search(
    method_name: str,
    method_runner,                 # callable: (gen_data, loop_time, **method_kwargs)->result_mat
    gen_data_grid: dict,           # e.g. {"n":[...], "p":[...], "k":[...], "alpha":[...], "beta_star":[...], "dat_type":[...]}
    method_kwargs_grid: dict,      # e.g. onlineFSA: {"eta":[0.01], "lbd":[0]}; Lasso: {"lbd_para":[np.array([...]), ...]}
    loop_time: int = 5,
    save_csv_path: str = None,
    verbose: bool = True,
):
    """
    Returns:
      df_results with columns:
        - gen_data fields
        - method hyper-params
        - DR, PCD, RMSE_mean, Runtime_mean
    """

    # 展开 gen_data 网格
    gen_keys = list(gen_data_grid.keys())
    gen_values = [gen_data_grid[k] if isinstance(gen_data_grid[k], (list, tuple)) else [gen_data_grid[k]] for k in gen_keys]

    # 展开 method kwargs 网格
    m_keys = list(method_kwargs_grid.keys())
    m_values = [method_kwargs_grid[k] if isinstance(method_kwargs_grid[k], (list, tuple)) else [method_kwargs_grid[k]] for k in m_keys]

    records = []
    total = 0

    for gen_combo in itertools.product(*gen_values):
        gen_data = dict(zip(gen_keys, gen_combo))
        k_true = int(gen_data["k"])

        for m_combo in itertools.product(*m_values):
            method_kwargs = dict(zip(m_keys, m_combo))

            # 跑实验
            t0 = time.time()
            result_mat = method_runner(gen_data=gen_data, loop_time=loop_time, **method_kwargs)
            t1 = time.time()

            # 汇总指标（以 result_mat 为准，t1-t0 只是额外 wall time）
            metrics = summarize_result_mat(result_mat, k_true=k_true)

            # 记录
            rec = {
                "method": method_name,
                **gen_data,
                **{f"hp_{k}": v for k, v in method_kwargs.items()},
                "loop_time": loop_time,
                **metrics,
                "wall_time": float(t1 - t0),
            }
            records.append(rec)
            total += 1

            if verbose:
                hp_str = ", ".join([f"{k}={v}" for k, v in method_kwargs.items()])
                print(
                    f"[done] {method_name} | {hp_str} | "
                    f"n={gen_data.get('n')} p={gen_data.get('p')} k={k_true} beta_star={gen_data.get('beta_star')} | "
                    f"DR={metrics['DR']}/{loop_time}, PCD={metrics['PCD']:.3f}, "
                    f"RMSE={metrics['RMSE_mean']:.4f}, time={metrics['Runtime_mean']:.4f}s"
                )

    df = pd.DataFrame(records)
    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)
        if verbose:
            print(f"Saved results to: {save_csv_path} (rows={len(df)})")
    return df


# =========================
# 4) 一个更“用户友好”的总入口：一次性支持多方法
# =========================
def grid_search_methods(
    methods: dict,
    gen_data_grid: dict,
    loop_time: int = 10,
    save_csv_path: str = None,
):
    """
    methods 示例：
    methods = {
      "onlineFSA": {
         "runner": run_onlineFSA,
         "kwargs_grid": {"eta":[0.01], "lbd":[0]}
      },
      "Lasso": {
         "runner": run_lasso,
         "kwargs_grid": {"lbd_para":[np.linspace(0.001,0.1,20)]}
      }
    }
    """
    dfs = []
    for name, cfg in tqdm(methods.items()):
        df = run_grid_search(
            method_name=name,
            method_runner=cfg["runner"],
            gen_data_grid=gen_data_grid,
            method_kwargs_grid=cfg.get("kwargs_grid", {}),
            loop_time=loop_time,
            save_csv_path=None,   # 先不单独存
            verbose=True,
        )
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    if save_csv_path is not None:
        df_all.to_csv(save_csv_path, index=False)
        print(f"[ALL] Saved combined results to: {save_csv_path} (rows={len(df_all)})")

    return df_all


# =========================
# 5) Example usage
# =========================
if __name__ == "__main__":

    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # -------- gen_data 网格（你给的组合）--------
    gen_data_grid = {
        "n": [500, 1000, 5000, 10000, 50000],
        "p": [1000],
        "k": [100],
        "alpha": [1],
        "beta_star": [1, 0.1, 0.01],
        "dat_type": [1],
    }

    # -------- 方法配置 --------
    # Lasso 的 lbd_para 你需要按你 Lassofeaturesel.Lasso_feature_sel 的接口来传
    # 常见是一个 lambda grid，比如 np.logspace(-4, 0, 40)
    lbd_para_range = {"start": -10, "end": 10}

    methods = {
        "onlineFSA": {
            "runner": run_onlineFSA,
            "kwargs_grid": {"eta": [0.01], "lbd": [0]},
        },
        "Lasso": {
            "runner": run_lasso,
            "kwargs_grid": {
                "lbd_para": [lbd_para_range],  # 传 dict（范围）
            },
        },
    }

    
    save_file = os.path.join(save_dir, "grid_results_onlineFSA_and_Lasso.csv")
    # for online methods
    df_all = grid_search_methods(
        methods=methods,
        gen_data_grid=gen_data_grid,
        loop_time=5,   # 按你的要求
        save_csv_path=save_file,
    )

    # for offline methods

    # print(df_all.head(1))



