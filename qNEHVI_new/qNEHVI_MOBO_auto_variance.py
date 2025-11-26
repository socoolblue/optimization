"""
초기 데이터에도 반복 데이터 포함_q는 모두반복하여 분산적용_
"실전이면 이거사용하라"

반복 측정한 샘플들은 각각 자기 분산을 노이즈로 사용함

단일 측정 샘플의 분산(노이즈)은 한 번이라도 반복 측정된 샘플이 있으면 그들의 평균 분산을 사용

모든 샘플이 단일 측정이면 → default_noise (0.01) 사용
--> 다만 이런 경우는 만들지 말고
이때는 'qNEHVI_MOBO_code_GPU_한번씩 시행하는 코드'를 이용하라 이 코드는 모든 샘플의 통일된 노이즈(분산을) 학습시킨다

==========================================================
   qNEHVI-based Multi-objective Bayesian Optimization (auto variance detection)
   ✅ y_data만 입력 (3n 길이 → n회 측정)
   ✅ 각 샘플별 측정 횟수 자동 인식 후 평균·분산 계산
   ✅ FixedNoiseGP 기반 모델
   ✅ Hypervolume 계산 및 인쇄
==========================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import torch
import numpy as np
import pandas as pd
from torch.optim import Adam

# ==========================================================
# FIXEDNOISEGP IMPORT (BoTorch 버전별 대응)
# ==========================================================
try:
    from botorch.models import FixedNoiseGP  # 0.16 이상
except ImportError:
    try:
        from botorch.models.gp_regression_fixed import FixedNoiseGP  # 0.15.0~0.15.1
    except ImportError:
        from botorch.models.gp_regression import SingleTaskGP
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
        class FixedNoiseGP(SingleTaskGP):
            """Fallback: older BoTorch builds"""
            def __init__(self, train_X, train_Y, train_Yvar, **kwargs):
                likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar)
                super().__init__(train_X, train_Y, likelihood=likelihood, **kwargs)

from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.initializers import gen_batch_initial_conditions
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# ==========================================================
# CONFIGURATION
# ==========================================================
config = {
    "num_objectives": 3,
    "num_variables": 12,
    "maximize": [True, True, True],
    "bounds": [
    (-8.219935292911435, 4.8678405666098925),
    (-4.529524136731765, 8.491828249116784),
    (-4.485586547153938, 7.464671202787634),
    (-4.678683041062727, 4.10771660957421),
    (-3.1486780009041593, 4.743911725110442),
    (-3.8200012211312333, 4.238666876454304),
    (-3.6859282450412456, 3.6804077253203857),
    (-3.8340695577663375, 3.600410804950003),
    (-3.7403453350118108, 3.2409026104937992),
    (-2.6439237960579014, 3.584396130267221),
    (-2.6281305042188903, 2.895901685121338),
    (-3.5432124601001793, 2.4775653447131045)
    ],
    "batch_q": 4,
    "num_restarts": 10,
    "raw_samples": 512,
    "random_seed": 42,
    "default_noise": 0.0001
}

torch.manual_seed(config["random_seed"])
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}, dtype: {dtype}")


# ==========================================================
# AUTO-DETECT REPETITIONS, COMPUTE MEAN & VAR
# ==========================================================
# 1차 루프: 반복 샘플들의 분산 먼저 수집
Y_mean_list, Y_var_list, all_repeated_vars = [], [], []

for i, ylist in enumerate(y_data):
    y = np.array(ylist)
    n_meas = len(y) // config["num_objectives"]
    y = y.reshape(n_meas, config["num_objectives"])

    if n_meas > 1:
        var = np.var(y, axis=0, ddof=1)
        all_repeated_vars.append(var)

# 🔸 반복 측정 분산들의 전체 평균
if len(all_repeated_vars) > 0:
    global_mean_var = np.mean(np.vstack(all_repeated_vars), axis=0)
    print(f"🌐 Global mean variance from repeated samples: {global_mean_var}")
else:
    global_mean_var = np.ones(config["num_objectives"]) * config["default_noise"]
    print(f"⚪ No repeated samples → using default noise: {global_mean_var}")

# 2차 루프: 평균/분산 최종 계산
for i, ylist in enumerate(y_data):
    y = np.array(ylist)
    n_meas = len(y) // config["num_objectives"]
    y = y.reshape(n_meas, config["num_objectives"])

    if n_meas == 1:
        mean = y[0]
        var = global_mean_var  # ✅ 실제 반복 측정 평균 분산으로 대체
        print(f"⚪ Sample {i}: 1 measurement → global mean variance applied")
    else:
        mean = np.mean(y, axis=0)
        var = np.var(y, axis=0, ddof=1)
        print(f"🔵 Sample {i}: {n_meas} measurements → variance computed")

    Y_mean_list.append(mean)
    Y_var_list.append(np.maximum(var, 1e-6))

X = torch.tensor(x_data, dtype=dtype, device=device)
Y = torch.tensor(np.array(Y_mean_list), dtype=dtype, device=device)
Yvar = torch.tensor(np.array(Y_var_list), dtype=dtype, device=device)

for i, maximize in enumerate(config["maximize"]):
    if not maximize:
        Y[:, i] = -Y[:, i]

print(f"✅ Data loaded: X={X.shape}, Y={Y.shape}, Yvar={Yvar.shape}")

# ==========================================================
# MODEL
# ==========================================================
model = FixedNoiseGP(
    train_X=X,
    train_Y=Y,
    train_Yvar=Yvar,
    covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=config["num_variables"])),
    input_transform=Normalize(config["num_variables"]),
    outcome_transform=Standardize(config["num_objectives"]),
).to(device=device, dtype=dtype)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
opt = Adam(mll.parameters(), lr=0.01)
for _ in range(200):
    opt.zero_grad()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets).sum()
    loss.backward()
    opt.step()
print("✅ GP MLE optimization finished.")

# ==========================================================
# REFERENCE POINT
# ==========================================================
with torch.no_grad():
    model.eval()
    mean = model.posterior(X).mean
    Y_adj = mean.clone()
    for i, maximize in enumerate(config["maximize"]):
        if not maximize:
            Y_adj[:, i] = -Y_adj[:, i]
    ref_point = (Y_adj.min(0).values - 2.0).clamp(min=-10.0)
    ref_point = torch.where(torch.isnan(ref_point),
                            torch.tensor(-1.0, dtype=dtype, device=device),
                            ref_point)
print(f"📍 Safe ref_point: {ref_point.tolist()}")

# ==========================================================
# HYPERVOLUME CALCULATION
# ==========================================================
with torch.no_grad():
    mask = is_non_dominated(Y)
    pareto_front = Y[mask]
    hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
    print(f"🌈 Hypervolume (current data): {float(hv):.4f}")

# ==========================================================
# ACQUISITION FUNCTION
# ==========================================================
partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y_adj)
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
acq_func = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point.tolist(),
    X_baseline=X,
    objective=IdentityMCMultiOutputObjective(),
    prune_baseline=True,
    sampler=sampler,
)

# ==========================================================
# ACQUISITION OPTIMIZATION
# ==========================================================
def optimize_acqf_torch(acqf, bounds, q, num_restarts, raw_samples, steps=100):
    bounds_t = torch.tensor(bounds, dtype=dtype, device=device).T
    Xraw = gen_batch_initial_conditions(acq_function=acqf, bounds=bounds_t, q=q,
                                        num_restarts=num_restarts, raw_samples=raw_samples).to(device)
    Xraw.requires_grad_(True)
    opt = Adam([Xraw], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = -acqf(Xraw).sum()
        loss.backward()
        opt.step()
        Xraw.data.clamp_(bounds_t[0], bounds_t[1])
    with torch.no_grad():
        vals = acqf(Xraw)
        best = torch.argmax(vals)
        return Xraw[best].detach(), vals[best].detach()

candidate, acq_value = optimize_acqf_torch(
    acq_func, config["bounds"], q=config["batch_q"],
    num_restarts=config["num_restarts"], raw_samples=config["raw_samples"], steps=150,
)

# ==========================================================
# OUTPUT
# ==========================================================
df_out = pd.DataFrame(candidate.cpu().numpy(),
                      columns=[f"x{i+1}" for i in range(config["num_variables"])])
print("\n===== Next Suggested Candidates =====")
print(df_out.to_string(index=False))
print(f"\n📈 Acquisition Value: {acq_value.item():.6f}")
