#Loading Libraries
import numpy as np
import torch

import os
from os import listdir
from os.path import isfile, join
from PIL import Image

import torch.optim as optim
from torch.autograd import grad
import time

import torch.utils.data as data_utils
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import Adam, lr_scheduler

from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from scipy.io import loadmat
from scipy.io import savemat

import torch
import onlineFSA
from sklearn.metrics import roc_curve, auc
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
nx=144

import clip
print(clip.available_models())
model, preprocess = clip.load('RN50x4', device)

relu = torch.nn.functional.relu
def features(net, x):
    x = x.type(net.conv1.weight.dtype)
    for conv, bn in [(net.conv1, net.bn1), (net.conv2, net.bn2), (net.conv3, net.bn3)]:
        x = relu(bn(conv(x)))
    x = net.avgpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    x=net.avgpool(x) 
    x=net.avgpool(x)
    return x

# 
def generate_features(dataset, model, mode):
    labels = torch.empty(0).cpu()
    i = 0
    for images, labs in tqdm(DataLoader(dataset, batch_size=100), desc=mode):
            images = images.to(device)
            labs = labs.cpu()
            with torch.no_grad():
                f = features(model.visual,images) 
                # print(f"f.shape = {f.shape}")
                # print(f"f.dtype = {f.dtype}")
                f_sq = f.squeeze(3).squeeze(2)
                # print(f"after")
                # print(f"f_sq.shape = {f_sq.shape}")
                # print(f"f_sq.dtype = {f_sq.dtype}")
                # assert 1 == 2

            f_sq=f_sq.cpu()
            if i==0:
                d=f_sq.shape[1]
                feat = torch.empty(0, d).cpu()
            feat = torch.cat((feat,f_sq),dim=0)
            labels = torch.cat((labels , labs),dim=0)
            i = i+1
    return feat, labels


# features_train, labels_train = generate_features(cifar100_train, model, mode='train')
# features_test, labels_test = generate_features(cifar100_test, model, mode='test')


# Save features to mat files
train_dir = r'../data/Cifar100/features/Clip/train/' 
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok=True)

# for i in range(500):
#     train_name = os.path.join(train_dir, f'{i}.mat')
#     x_train = features_train[labels_train == i,:]
#     savemat(train_name, {'feature': x_train.float().numpy()})

test_dir = r'../data/Cifar100/features/Clip/val'
if not os.path.exists(test_dir):
    os.makedirs(test_dir, exist_ok=True)

# for i in range(100):
#     test_name = os.path.join(test_dir, f'{i}.mat')
#     x_test = features_test[labels_test == i,:]
#     savemat(test_name, {'feature': x_test.float().numpy()})



def build_binary_cifar_features(
    trFeatures, trY, valFeatures, valY,
    cls_a=0, cls_b=1,
    device=device
):
    """
    从 CIFAR-100 中抽取两个类别 cls_a / cls_b 做二分类，
    标签映射为 +1 / -1，并把所有 tensor 放到 GPU 上。
    """
    # 训练集
    mask_tr = (trY == cls_a) | (trY == cls_b)
    Xtr = trFeatures[mask_tr].to(device)              # [n_tr, d]
    ytr = trY[mask_tr].to(device)                     # [n_tr]
    ytr_bin = torch.where(ytr == cls_a,
                          torch.tensor(1., device=device),
                          torch.tensor(-1., device=device))

    # 验证集
    mask_val = (valY == cls_a) | (valY == cls_b)
    Xval = valFeatures[mask_val].to(device)           # [n_val, d]
    yval = valY[mask_val].to(device)
    yval_bin = torch.where(yval == cls_a,
                           torch.tensor(1., device=device),
                           torch.tensor(-1., device=device))

    return Xtr, ytr_bin, Xval, yval_bin



def build_running_aves_from_numpy(X_np, y_np, batch_size):
    """
    给定整套训练数据 (X_np, y_np)，按 mini-batch 累加 running averages，
    输出格式与原始 OFSA 代码中的 ra_sum 完全一致。

    参数
    ----
    X_np : np.ndarray, shape (n, p), float32
    y_np : np.ndarray, shape (n,) 或 (n,1)，float32，取值通常为 ±1
    batch_size : int

    返回
    ----
    ra_sum : dict
        {
          "n":  样本总数,
          "Sx": (1, p),
          "Sy": 标量,
          "Sxx": (p, p),
          "Sxy": (p, 1),
          "Syy": 标量
        }
        可以直接喂给 onlineFSA.standardize_ra
    """
    datat = np.float32

    # 类型整理
    X_np = X_np.astype(datat)
    y_np = y_np.astype(datat).reshape(-1, 1)   # [n, 1]

    n, p = X_np.shape
    # 和原代码一样的写法是 int(n / batch_size)，
    # 为了不丢最后一小段，这里用 ceil；如果你想完全对齐原实现可以改成 int(n / batch_size)
    num_batch = int(np.ceil(n / batch_size))

    # 初始化 running averages（和原 OFSA_feasel 完全一致）
    n_sum   = 0
    Sx_sum  = np.zeros((1, p), dtype=datat)
    Sy_sum  = 0.0
    Sxx_sum = np.zeros((p, p), dtype=datat)
    Sxy_sum = np.zeros((p, 1), dtype=datat)
    Syy_sum = 0.0

    ra_sum = {
        "n":   n_sum,
        "Sx":  Sx_sum,
        "Sy":  Sy_sum,
        "Sxx": Sxx_sum,
        "Sxy": Sxy_sum,
        "Syy": Syy_sum,
    }

    # 按 batch 遍历真实数据，替代原来的 eqcorrdat_cls 生成过程
    for i in range(num_batch):
        start = i * batch_size
        end   = min((i + 1) * batch_size, n)
        if start >= end:
            break

        X_mb = X_np[start:end, :]      # [mb, p]
        y_mb = y_np[start:end, :]      # [mb, 1]

        # 原代码：ra_mb = onlineFSA.running_aves(Xtr_mb, Ytr_mb)
        ra_mb = onlineFSA.running_aves(X_mb, y_mb)

        # 原代码：ra_sum = onlineFSA.add_runningaves(ra_sum, ra_mb)
        ra_sum = onlineFSA.add_runningaves(ra_sum, ra_mb)

    return ra_sum


def OFSA_feasel_cifar(
    Xtr_gpu, ytr_gpu,
    Xval_gpu, yval_gpu,
    OFSA_para,
    batch_size: int,
):
    """
    针对 CIFAR-100 提取出来的特征，使用 OnlineFSA 做二分类特征选择 + 线性判别。

    参数
    ----
    Xtr_gpu : torch.Tensor, shape (n_train, p)，在 GPU 上
    ytr_gpu : torch.Tensor, shape (n_train,)，在 GPU 上，标签为 ±1
    Xval_gpu: torch.Tensor, shape (n_val, p)，在 GPU 上
    yval_gpu: torch.Tensor, shape (n_val,)，在 GPU 上，标签为 ±1
    OFSA_para : dict, 包含 { "k", "eta", "mu", "lbd", "N_iter", "pretr_time" }
    batch_size: int, 构造 running averages 时的 mini-batch 大小

    返回
    ----
    sel       : np.ndarray, 被选中的特征下标（长度为 k）
    roc_auc   : float, 在验证集上的 AUC
    time_cost : float, onlineFSA 求解时间（秒）
    """

    # ========= 1. 从 GPU Tensor 拿到 numpy =========
    Xtr_np = Xtr_gpu.detach().cpu().numpy().astype(np.float32)
    ytr_np = ytr_gpu.detach().cpu().numpy().astype(np.float32)

    n, p = Xtr_np.shape
    k          = OFSA_para["k"]
    eta        = OFSA_para["eta"]
    mu         = OFSA_para["mu"]
    lbd        = OFSA_para["lbd"]
    N_iter     = OFSA_para["N_iter"]
    pretr_time = OFSA_para["pretr_time"]

    # ========= 2. 构造 running averages（模拟原 eqcorrdat 增量数据） =========
    ra_sum = build_running_aves_from_numpy(Xtr_np, ytr_np, batch_size)

    # 与原 OFSA_feasel 的 sanity check 一致
    if ra_sum["n"] != n:
        print(f"[WARN] running averages n={ra_sum['n']} 与实际样本数 n={n} 不一致")

    # ========= 3. 标准化 running averages =========
    XX_normalize, XY_normalize, mu_x, mu_y, std_x = onlineFSA.standardize_ra(ra_sum)

    # ========= 4. 调用 onlineFSA 进行特征选择 =========
    FSA_para = {
        "n": n,
        "k": k,
        "eta": eta,
        "mu": mu,
        "lbd": lbd,
        "N_iter": N_iter,
    }

    t_start = time.process_time()
    beta_sel, sel = onlineFSA.onlineFSA(XX_normalize, XY_normalize, FSA_para, pretr_time)
    t_end = time.process_time()
    time_cost = t_end - t_start

    sel = np.array(sel, dtype=int)  # 确保是 numpy int 数组

    # ========= 5. 在验证集上评估 =========
    Xval_np = Xval_gpu.detach().cpu().numpy().astype(np.float32)
    yval_np = yval_gpu.detach().cpu().numpy().astype(np.float32)  # 取值为 ±1

    n_val = Xval_np.shape[0]

    # 和原版一样，用训练的均值 / 方差来标准化验证集
    Xval_std = Xval_np - np.ones((n_val, 1), dtype=np.float32).dot(mu_x)
    inv_sigma = 1.0 / std_x
    Xval_std = Xval_std * inv_sigma  # 广播到 (n_val, p)

    # 将 beta_sel 填回完整维度向量
    beta_OFSA = np.zeros((p, 1), dtype=np.float32)
    beta_OFSA[sel] = beta_sel

    # 线性打分
    Yscore_val = Xval_std.dot(beta_OFSA).ravel()  # shape (n_val,)

    # 由于 y ∈ {+1, -1}，指定 pos_label=1.0
    fpr, tpr, thresholds = roc_curve(yval_np, Yscore_val, pos_label=1.0)
    roc_auc = auc(fpr, tpr)

    return sel, roc_auc, time_cost



def OFSA_numexp_cifar(
    Xtr_gpu, ytr_gpu,
    Xval_gpu, yval_gpu,
    OFSA_para,
    mb_size: int,
    exp_times: int = 10,
):
    """
    多次重复 OFSA_feasel_cifar，方便统计均值方差。
    这里我们只记录 seed, AUC, time 三列（真实数据上没有“真稀疏支撑”，
    所以不再计算 DR/PCD）。
    """
    n_tr = Xtr_gpu.shape[0]
    results = np.zeros((exp_times, 3))   # seed, AUC, time

    for i in range(exp_times):
        seed = 2026 + i
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 打乱训练顺序（仍在 GPU 上完成）
        perm = torch.randperm(n_tr, device=Xtr_gpu.device)
        Xtr_shuf = Xtr_gpu[perm]
        ytr_shuf = ytr_gpu[perm]

        sel, roc_auc, t_cost = OFSA_feasel_cifar(
            Xtr_shuf, ytr_shuf,
            Xval_gpu, yval_gpu,
            OFSA_para,
            batch_size=mb_size,
        )

        results[i, 0] = seed
        results[i, 1] = roc_auc
        results[i, 2] = t_cost

        print(f"[Run {i}] seed={seed}, AUC={roc_auc:.4f}, time={t_cost:.3f}s, |sel|={len(sel)}")

    return results


#Loading data
def load_data(path, file):
    name=os.path.join(path, file)
    m=loadmat(name)
    x=torch.tensor(m['feature'])
    return x.float()


if __name__ == '__main__':

    train_dir = r'../data/Cifar100/features/Clip/train/' 
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)

    test_dir = r'../data/Cifar100/features/Clip/val'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    nc = 100 #number of classes
    d = 2560 #number of features

    
    trFeatures = torch.empty((0,d))
    trY  = torch.empty(0)

    for j in range(0,nc):
        x = load_data(train_dir,'{}.mat'.format(j))
        rep = x.shape[0]
        y =  torch.tensor(j)
        y1 = y.repeat(rep)
        trFeatures = torch.cat((trFeatures,x), dim = 0)
        trY = torch.cat((trY,y1),dim=0) 

    valFeatures = torch.empty((0,d))
    valY  = torch.empty(0)
    for j in range(0,nc):
        x = load_data(test_dir,'{}.mat'.format(j))
        rep = x.shape[0]
        y =  torch.tensor(j)
        y1 = y.repeat(rep)
        valFeatures = torch.cat((valFeatures,x), dim = 0)
        valY = torch.cat((valY, y1),dim=0) 
    
    # 1) 把原始特征搬到 GPU（后面 build_binary_cifar 会再筛选）
    trFeatures = trFeatures.to(device)
    trY = trY.to(device)
    valFeatures = valFeatures.to(device)
    valY = valY.to(device)

    # 2) 任选两个 CIFAR-100 类别做二分类，比如 class 0 vs class 37
    cls_a, cls_b = 0, 37
    Xtr_bin, ytr_bin, Xval_bin, yval_bin = build_binary_cifar_features(
        trFeatures, trY, valFeatures, valY,
        cls_a=cls_a, cls_b=cls_b,
        device=device,
    )

    print("Train shape:", Xtr_bin.shape, ytr_bin.shape)
    print("Val   shape:", Xval_bin.shape, yval_bin.shape)

    # 3) 设置 OFSA 参数（k 是希望选取的特征个数）
    OFSA_para = {
        "k": 100,          # 例如选 200 个最重要的特征
        "eta": 0.01,
        "mu": 5,
        "lbd": 0.01,
        "N_iter": 200,
        "pretr_time": 30,
    }

    mb_size = 512
    exp_times = 10

    OFSA_results = OFSA_numexp_cifar(
        Xtr_bin, ytr_bin,
        Xval_bin, yval_bin,
        OFSA_para,
        mb_size=mb_size,
        exp_times=exp_times,
    )

    print("Mean AUC:", OFSA_results[:, 1].mean())
    print("Mean time:", OFSA_results[:, 2].mean())


