import os
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Lasso

from Lasso_num_exp import Lasso_num_exp
from onlineFSA_numerical_exp import onlineFSA_numexp

import datageneration 
import onlineFSA   # 这里假设你的 running_aves / standardize_ra / onlineFSA / OLS_runningaves 都在 onlineFSA.py 里
import Lassofeaturesel 

# -----------------------------
# 数据生成（复用你已有的 datageneration.generate_data）
# from datageneration import generate_data
# -----------------------------


def compute_regret_path_squared_loss(X, y, beta_seq, ridge=0.0):
    """
    计算真正的 online regret (需要完整 beta 序列):

      R_n = (1/n) sum_{i=1}^n f(beta_i; z_i)
            - min_beta (1/n) sum_{i=1}^n f(beta; z_i),

    其中 f(beta; z_i) = 0.5 * (y_i - x_i^T beta)^2.

    参数
    ----
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,) or (n,1)
    beta_seq : np.ndarray, shape (n, p)
        第 i 行是算法在时刻 i 使用的 beta_i
    ridge : float
        offline 最优解里加一点 L2 正则以避免 X^T X 奇异

    返回
    ----
    regret : float
    online_avg_loss : float
    offline_opt_loss : float
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    beta_seq = np.asarray(beta_seq, dtype=np.float64)

    n, p = X.shape
    assert beta_seq.shape == (n, p), "beta_seq 形状应为 (n, p)"

    # ---- 1) online 部分：每一步用自己的 beta_i ----
    y_hat_online = np.sum(X * beta_seq, axis=1)       # [n]
    err_online = y_hat_online - y
    online_avg_loss = 0.5 * np.mean(err_online ** 2)

    # ---- 2) offline 最优：最小化同一个平方损失 ----
    XtX = X.T @ X
    if ridge > 0.0:
        XtX = XtX + ridge * np.eye(p)
    Xty = X.T @ y
    beta_offline = np.linalg.solve(XtX, Xty)          # [p]

    y_hat_offline = X @ beta_offline
    err_offline = y_hat_offline - y
    offline_opt_loss = 0.5 * np.mean(err_offline ** 2)

    regret = online_avg_loss - offline_opt_loss
    return regret, online_avg_loss, offline_opt_loss


def offline_lasso_fit_select(
    X_train, y_train,
    lambda_grid,
    k_target=None,
    selection_mode="cv",   # "cv" or "closest_k"
    n_splits=5,
    random_state=123
):
    """
    Offline Lasso:
      - selection_mode="cv": 用 K-fold CV 选使 CV-MSE 最小的 lambda
      - selection_mode="closest_k": 在 lambda_grid 中选 |#nonzero - k_target| 最小的 lambda
    返回:
      lbd_sel, beta_lasso, sel_index, cv_mse_list (若 selection_mode="cv" 否则 None)
    """

    # 保证 y 是一维
    y_train = y_train.reshape(-1)

    if selection_mode == "cv":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_mse_list = []

        for lbd in lambda_grid:
            fold_mses = []
            for tr_idx, va_idx in kf.split(X_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                model = Lasso(alpha=lbd, fit_intercept=False, max_iter=10000)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_va)
                fold_mses.append(np.mean((y_va - pred) ** 2))

            cv_mse_list.append(np.mean(fold_mses))

        cv_mse_list = np.array(cv_mse_list)
        lbd_sel = float(lambda_grid[np.argmin(cv_mse_list)])

        model = Lasso(alpha=lbd_sel, fit_intercept=False, max_iter=10000)
        model.fit(X_train, y_train)
        beta_lasso = model.coef_
        sel_index = np.flatnonzero(beta_lasso)

        return lbd_sel, beta_lasso, sel_index, cv_mse_list

    elif selection_mode == "closest_k":
        if k_target is None:
            raise ValueError("selection_mode='closest_k' requires k_target.")

        best = None  # (gap, lbd, beta, sel)
        for lbd in lambda_grid:
            model = Lasso(alpha=lbd, fit_intercept=False, max_iter=10000)
            model.fit(X_train, y_train)
            beta = model.coef_
            sel = np.flatnonzero(beta)
            gap = abs(len(sel) - k_target)
            if best is None or gap < best[0]:
                best = (gap, float(lbd), beta, sel)

        _, lbd_sel, beta_lasso, sel_index = best
        return lbd_sel, beta_lasso, sel_index, None

    else:
        raise ValueError("selection_mode must be 'cv' or 'closest_k'.")


def offline_lasso_num_exp(gen_dat, lambda_grid, loop_time=5, selection_mode="cv", n_splits=5):
    """
    对标你的 Lasso_num_exp，但这里是 offline:
      - 每次循环生成一份训练集（全量可见）
      - 通过 CV(或 closest_k) 选择 lambda
      - 用所选特征 refit OLS，再在新生成的 test set 上算 RMSE
    result_mat columns:
      [seed, lbd_sel, num_true, rmse, time_cost]
    """
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]

    result_mat = np.zeros((loop_time, 5))

    for i in range(loop_time):
        seed = i + 100
        np.random.seed(seed)
        result_mat[i, 0] = seed

        # 1) 生成训练数据（离线：一次性拿到全部）
        X_tr, Y_tr, betastar_vec, istar = datageneration.generate_data(gen_dat)

        # 2) 标准化（和你原实现一致：X 标准化，Y 仅中心化）
        X_tr_scale = preprocessing.scale(X_tr)
        Y_tr_center = preprocessing.scale(Y_tr, with_std=False)  # shape (n,1) or (n,)

        # 3) Offline 选择 lambda + 拟合 Lasso
        t_start = time.process_time()
        lbd_sel, beta_lasso, sel_index, cv_mse = offline_lasso_fit_select(
            X_tr_scale,
            Y_tr_center,
            lambda_grid=lambda_grid,
            k_target=k,
            selection_mode=selection_mode,
            n_splits=n_splits,
            random_state=seed
        )
        t_end = time.process_time()
        time_cost = t_end - t_start

        result_mat[i, 1] = lbd_sel
        result_mat[i, 4] = time_cost

        # 4) DR/PCD 相关：命中真实变量数
        num_true_var = len(np.intersect1d(istar, sel_index))
        result_mat[i, 2] = num_true_var

        # 5) refit OLS（只在选中特征上）
        if len(sel_index) == 0:
            # 极端情况：没有选到任何特征
            rmse = np.nan
            result_mat[i, 3] = rmse
            continue

        X_tr_sel = X_tr_scale[:, sel_index]
        ols = LinearRegression(fit_intercept=False).fit(X_tr_sel, Y_tr_center.reshape(-1))
        beta_hat_ols = ols.coef_.reshape(-1, 1)  # shape (#sel,1)

        # 6) 生成测试数据（沿用你原逻辑：n=p 的测试集）
        gen_test = {"n": p, "p": p, "k": k, "alpha": alpha, "beta_star": beta_star, "dat_type": 1}
        X_test, Y_test, _, _ = datageneration.generate_data(gen_test)

        # 用训练集的 scaler 标准化测试集（与你的原实现对齐）
        X_scaler = preprocessing.StandardScaler().fit(X_tr)
        Y_mean = np.mean(Y_tr, axis=0)  # 用训练集均值中心化测试 Y

        testX_scale = X_scaler.transform(X_test)
        testX_sel = testX_scale[:, sel_index]
        testY_center = (Y_test - Y_mean).reshape(-1)

        # RMSE
        testY_hat = testX_sel.dot(beta_hat_ols).reshape(-1)
        err = testY_center - testY_hat
        rmse = np.sqrt(np.sum(err ** 2) / p)
        result_mat[i, 3] = rmse

    return result_mat


def offlineFSA_num_exp(gen_dat, eta, lbd, loop_time=5, pre_train=15, N_iter=200, mu=10):
    """
    Offline FSA numerical experiment (Lasso_num_exp-style)

    Parameters
    ----------
    gen_dat : dict
        {"n": n, "p": p, "k": k, "alpha": alpha, "beta_star": beta_star, "dat_type": dat_type}
    eta : float
        learning rate
    lbd : float
        shrinkage coefficient in gradient update (L2-like)
    loop_time : int
        number of repeated trials with different random seeds (default 5)
    pre_train : int
        pre-train steps in FSA loop_list = [-pre_train,...,N_iter]
    N_iter : int
        number of iterations in FSA
    mu : float
        annealing parameter in M_e schedule

    Returns
    -------
    result_mat : np.ndarray, shape (loop_time, 5)
        columns: [seed, eta, num_true_var, rmse, time_cost]
    """
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    dat_type = gen_dat["dat_type"]

    result_mat = np.zeros((loop_time, 5))

    for i in range(loop_time):
        seed = i + 100
        np.random.seed(seed)

        result_mat[i, 0] = seed
        result_mat[i, 1] = eta

        # -------------------------
        # 1) Offline generate full training data
        # -------------------------
        X_tr, Y_tr, beta_vec, istar = datageneration.generate_data(gen_dat)
        # Y_tr expected shape: (n,1)

        # -------------------------
        # 2) Build running averages from full data (offline)
        # -------------------------
        rs = onlineFSA.running_aves(X_tr, Y_tr)

        # -------------------------
        # 3) Standardize running averages -> XX_normalize, XY_normalize
        # -------------------------
        XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(rs)

        # -------------------------
        # 4) Train FSA (offline, using full-data XX/XY)
        # -------------------------
        FSA_para = {"n": n, "k": k, "eta": eta, "mu": mu, "lbd": lbd, "N_iter": N_iter}

        t_start = time.process_time()
        beta_sel, sel = onlineFSA.onlineFSA(XX_normalize, XY_normalize, FSA_para, pre_train)
        t_end = time.process_time()
        time_cost = t_end - t_start

        result_mat[i, 4] = time_cost

        # -------------------------
        # 5) DR / PCD part: number of true selected
        # -------------------------
        num_true_var = len(np.intersect1d(istar, sel))
        result_mat[i, 2] = num_true_var

        # -------------------------
        # 6) Refit by OLS on selected variables (still using running sums form)
        # -------------------------
        if len(sel) == 0:
            result_mat[i, 3] = np.nan
            continue

        XX_sel = XX_normalize[np.ix_(sel, sel)]
        XY_sel = XY_normalize[sel]
        beta_ols = onlineFSA.OLS_runningaves(XX_sel, XY_sel, 0.0)  # no ridge in refit

        beta_hat = np.zeros((p, 1))
        beta_hat[sel] = beta_ols

        # -------------------------
        # 7) Generate test data (keep same style as your onlineFSA_numexp)
        #    常见写法：n_test = p 或 n_test = n；这里给一个可选策略：n_test = min(n,p)
        # -------------------------
        n_test = min(n, p)
        gen_test = {"n": n_test, "p": p, "k": k, "alpha": alpha, "beta_star": beta_star, "dat_type": dat_type}
        X_te, Y_te, _, _ = datageneration.generate_data(gen_test)

        # -------------------------
        # 8) Standardize test data using training statistics (mu_x, mu_y, std_x)
        # -------------------------
        testY_center = Y_te - mu_y * np.ones((n_test, 1))
        testX_center = X_te - np.ones((n_test, 1)).dot(mu_x)

        inv_sigma = 1.0 / std_x
        testX_std = inv_sigma * testX_center  # broadcast

        # -------------------------
        # 9) RMSE
        # -------------------------
        testY_hat = testX_std.dot(beta_hat)
        err = testY_center - testY_hat
        rmse = np.sqrt(np.sum(err ** 2) / n_test)
        result_mat[i, 3] = rmse

    return result_mat


# -----------------------------
# 1) Offline Lasso (你已有 Lasso_num_exp 可直接用；这里保留接口兼容)
# -----------------------------
def offlineLasso_num_exp(gen_dat, lbd_para, loop_time=5):
    # 直接复用你已有的 Lasso_num_exp
    return Lasso_num_exp(gen_dat, lbd_para, loop_time)


# -----------------------------
# 2) Online Lasso (模拟版：流式接收数据 -> 拼接 -> 做一次 Lasso)
#    若你有真正 online lasso，可直接替换这个函数实现。
# -----------------------------
def onlineLasso_num_exp(gen_dat, lbd_para, loop_time=5, batch_size=None):
    """
    Simulated online lasso:
      - receive data in mini-batches
      - concatenate to full train set
      - run lasso once (same as offline lasso objective), then refit OLS and test
    Output format aligned with Lasso_num_exp: (loop_time, 5)
      col0 seed, col1 chosen_lambda, col2 num_true, col3 rmse, col4 time_cost
    """
    n = gen_dat["n"]
    p = gen_dat["p"]
    k = gen_dat["k"]
    alpha = gen_dat["alpha"]
    beta_star = gen_dat["beta_star"]
    dat_type = gen_dat["dat_type"]

    if batch_size is None:
        batch_size = min(p, n)  # 对齐你 onlineFSA 的 batch 逻辑
    num_batches = int(np.ceil(n / batch_size))

    result_mat = np.zeros((loop_time, 5))

    for i in range(loop_time):
        seed = i + 100
        np.random.seed(seed)
        result_mat[i, 0] = seed

        # --- stream receive ---
        X_list, Y_list = [], []
        istar_ref = None
        for _ in range(num_batches):
            cur_n = min(batch_size, n - len(X_list) * batch_size)
            if cur_n <= 0:
                break
            gen_partial = {**gen_dat, "n": cur_n}
            Xb, Yb, _, istar = datageneration.generate_data(gen_partial)
            X_list.append(Xb)
            Y_list.append(Yb)
            istar_ref = istar

        X_tr = np.vstack(X_list)
        Y_tr = np.vstack(Y_list)

        # --- standardize like your offline Lasso code ---
        X_tr_scale = preprocessing.scale(X_tr)
        Y_tr_center = preprocessing.scale(Y_tr, with_std=False)

        # --- lasso selection (same as your Lassofeaturesel.Lasso_feature_sel) ---
        t_start = time.process_time()
        lasso_sel_index, lbd_sel = Lassofeaturesel.Lasso_feature_sel(
            X_tr_scale, Y_tr_center, lbd_para, k
        )
        t_end = time.process_time()

        time_cost = t_end - t_start
        result_mat[i, 1] = lbd_sel
        result_mat[i, 4] = time_cost

        num_true_var = len(np.intersect1d(istar_ref, lasso_sel_index))
        result_mat[i, 2] = num_true_var

        # --- refit OLS ---
        if len(lasso_sel_index) == 0:
            result_mat[i, 3] = np.nan
            continue

        X_tr_sel = X_tr_scale[:, lasso_sel_index]
        OLS_fit = LinearRegression(fit_intercept=False).fit(X_tr_sel, Y_tr_center)
        beta_hat_ols = OLS_fit.coef_.reshape(-1, 1)

        # --- test data (保持与 Lasso_num_exp 一致：n_test = p) ---
        gen_test = {"n": p, "p": p, "k": k, "alpha": alpha, "beta_star": beta_star, "dat_type": 1}
        X_test, Y_test, _, _ = datageneration.generate_data(gen_test)

        X_scaler = preprocessing.StandardScaler().fit(X_tr)
        Y_mean = np.mean(Y_tr, axis=0)

        testX_scale = X_scaler.transform(X_test)[:, lasso_sel_index]
        testY_center = (Y_test - Y_mean.reshape(1, -1)).reshape(-1, 1)

        testY_hat = testX_scale.dot(beta_hat_ols)
        err = testY_center - testY_hat
        rmse = np.sqrt(np.sum(err ** 2) / p)
        result_mat[i, 3] = rmse

    return result_mat


# -----------------------------
# 3) 统一的 “一次实验调用” wrapper
# -----------------------------
def run_one_setting(method, gen_data, loop_time, eta=None, lbd=None, lbd_para=None, **kwargs):
    """
    Returns result_mat (loop_time, 5)
    """
    if method == "onlineFSA":
        if eta is None or lbd is None:
            raise ValueError("onlineFSA requires eta and lbd.")
        return onlineFSA_numexp(gen_data, eta, lbd, loop_time)

    if method == "offlineFSA":
        if eta is None or lbd is None:
            raise ValueError("offlineFSA requires eta and lbd.")
        return offlineFSA_num_exp(gen_data, eta=eta, lbd=lbd, loop_time=loop_time, **kwargs)

    if method == "offlineLasso":
        if lbd_para is None:
            raise ValueError("offlineLasso requires lbd_para, e.g., {'start':-10,'end':10}.")
        return offlineLasso_num_exp(gen_data, lbd_para=lbd_para, loop_time=loop_time)

    if method == "onlineLasso":
        if lbd_para is None:
            raise ValueError("onlineLasso requires lbd_para, e.g., {'start':-10,'end':10}.")
        return onlineLasso_num_exp(gen_data, lbd_para=lbd_para, loop_time=loop_time, **kwargs)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# 4) 统一 grid search 主入口
# -----------------------------
def grid_search_methods(
    method,
    n_list,
    p,
    k,
    alpha,
    beta_star_list,
    dat_type=1,
    loop_time=5,
    # FSA params
    eta=None,
    lbd=None,
    # Lasso params
    lbd_para=None,
    # output
    save_csv_path=None,
    # pass-through for method-specific extra args
    **kwargs
):
    """
    Unified pluggable runner.

    Returns
    -------
    df_results : pd.DataFrame
      columns include:
        method, n, p, k, alpha, beta_star, dat_type, loop_time,
        DR_mean, DR_std, PCD_mean, PCD_std, RMSE_mean, RMSE_std, Time_mean, Time_std,
        plus method-specific columns (e.g., eta, lbd, lbd_start/end).
    """
    rows = []

    for n in tqdm(n_list):
        for beta_star in beta_star_list:
            gen_data = {
                "n": int(n),
                "p": int(p),
                "k": int(k),
                "alpha": float(alpha),
                "beta_star": float(beta_star),
                "dat_type": int(dat_type),
            }

            result_mat = run_one_setting(
                method=method,
                gen_data=gen_data,
                loop_time=loop_time,
                eta=eta,
                lbd=lbd,
                lbd_para=lbd_para,
                **kwargs
            )
            # result_mat columns (common):
            # col2: num_true_var, col3: rmse, col4: time_cost
            num_true = result_mat[:, 2]
            rmse = result_mat[:, 3]
            tcost = result_mat[:, 4]

            DR = (num_true == k).astype(float)           # exact recovery indicator
            PCD = num_true / float(k)                    # proportion correctly discovered

            row = {
                "method": method,
                "n": n,
                "p": p,
                "k": k,
                "alpha": alpha,
                "beta_star": beta_star,
                "dat_type": dat_type,
                "loop_time": loop_time,

                "DR_mean": DR.mean(),
                "DR_std": DR.std(ddof=1) if loop_time > 1 else 0.0,

                "PCD_mean": PCD.mean(),
                "PCD_std": PCD.std(ddof=1) if loop_time > 1 else 0.0,

                "RMSE_mean": np.nanmean(rmse),
                "RMSE_std": np.nanstd(rmse, ddof=1) if loop_time > 1 else 0.0,

                "Time_mean": tcost.mean(),
                "Time_std": tcost.std(ddof=1) if loop_time > 1 else 0.0,
            }

            # attach method-specific hyperparams for logging
            if method in ("onlineFSA", "offlineFSA"):
                row["eta"] = eta
                row["lbd"] = lbd
            if method in ("offlineLasso", "onlineLasso"):
                row["lbd_start"] = lbd_para["start"]
                row["lbd_end"] = lbd_para["end"]

            rows.append(row)

    df_results = pd.DataFrame(rows)

    if save_csv_path is not None:
        df_results.to_csv(save_csv_path, index=False)

    return df_results


# -----------------------------
# 5) Example usage
# -----------------------------
if __name__ == "__main__":
    n_list = [500, 1000, 5000, 10000, 50000, 100000]
    beta_star_list = [1.0, 0.1, 0.01]

    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fsa_file = os.path.join(save_dir, "offlineFSA_grid.csv")
    # FSA grid
    df_fsa = grid_search_methods(
        method="offlineFSA",
        n_list=n_list,
        p=1000,
        k=100,
        alpha=1,
        beta_star_list=beta_star_list,
        dat_type=1,
        eta=0.01,
        lbd=0.0,
        loop_time=5,
        save_csv_path=fsa_file,
        # 这些是 offlineFSA_num_exp 支持的额外参数（如果你用了我的版本）
        pre_train=15,
        N_iter=200,
        mu=10
    )
    print(df_fsa.head(1))


    lasso_file = os.path.join(save_dir, "offlineLasso_grid.csv")
    # Lasso grid
    df_lasso = grid_search_methods(
        method="offlineLasso",
        n_list=n_list,
        p=1000,
        k=100,
        alpha=1,
        beta_star_list=beta_star_list,
        dat_type=1,
        lbd_para={"start": -10, "end": 10},
        loop_time=5,
        save_csv_path=lasso_file
    )
    print(df_lasso.head(1))
