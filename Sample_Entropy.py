# %%  environment setup
import os
import numpy as np
from numba import njit
from joblib import Parallel, delayed

# %% functions
@njit(fastmath=True)
def sample_entropy_numba(x, m, r):
    N = x.shape[0]
    A = 0
    B = 0

    for i in range(N - m):
        for j in range(i + 1, N - m):
            match_m = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    match_m = False
                    break

            if match_m:
                B += 1
                if abs(x[i + m] - x[j + m]) <= r:
                    A += 1

    if A == 0 or B == 0:
        return np.nan

    return -np.log(A / B)


@njit(fastmath=True)
def sampen_windowed_1d_numba(x, m, r_factor, win_len, step):
    N = x.shape[0]

    if N < win_len:
        return np.nan

    count = 0
    acc = 0.0

    for start in range(0, N - win_len + 1, step):
        seg = x[start:start + win_len]

        mean = 0.0
        for v in seg:
            mean += v
        mean /= win_len

        var = 0.0
        for v in seg:
            diff = v - mean
            var += diff * diff
        var /= win_len

        std = np.sqrt(var)
        if std == 0:
            continue

        r = r_factor * std
        val = sample_entropy_numba(seg, m, r)
        if not np.isnan(val):
            acc += val
            count += 1

    if count == 0:
        return np.nan

    return acc / count


def sampen_trial_channel_parallel(
    data_3d,
    m=2,
    r_factor=0.2,
    win_len=512,
    step=512,
    n_jobs=24   # 改为 8（9700X 最优）
):
    trial, _, channel = data_3d.shape
    tasks = [(tr, ch) for tr in range(trial) for ch in range(channel)]

    def _worker(tr, ch):
        x = data_3d[tr, :, ch].astype(np.float64)
        return sampen_windowed_1d_numba(
            x,
            m,
            r_factor,
            win_len,
            step
        )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(_worker)(tr, ch) for tr, ch in tasks
    )

    return np.array(results).reshape(trial, channel)


def inspect_sampen(name, data):
    print(f"\n{name}")
    print("shape:", data.shape)
    print("min :", np.nanmin(data))
    print("max :", np.nanmax(data))
    print("mean:", np.nanmean(data))
    print("std :", np.nanstd(data))


# %%  main function
def main():

    result_file = "sampen_results.npz"

    if os.path.exists(result_file):
        print(f"检测到已存在结果文件：{result_file}")
        print("跳过样本熵计算，直接进行数值自检...\n")

        res = np.load(result_file)

        rad_left_sampen  = res["rad_left_sampen"]
        tan_left_sampen  = res["tan_left_sampen"]
        rad_right_sampen = res["rad_right_sampen"]
        tan_right_sampen = res["tan_right_sampen"]
        rad_sampen = res["rad_sampen"] 
        tan_sampen = res["tan_sampen"]

        inspect_sampen("rad_left_sampen",  rad_left_sampen)
        inspect_sampen("tan_left_sampen",  tan_left_sampen)
        inspect_sampen("rad_right_sampen", rad_right_sampen)
        inspect_sampen("tan_right_sampen", tan_right_sampen)
        inspect_sampen("rad_sampen", rad_sampen) 
        inspect_sampen("tan_sampen", tan_sampen)

        print("\n数值自检完成，未进行重复计算。")
        return


    print("未检测到样本熵结果文件，开始计算...\n")

    # ---- 载入 trial 级 3D 数据 ----
    # data = np.load("trial_data_3d.npz")
    data = np.load("trial_data_3d.npz")

    rad_left_3d  = data["rad_left_3d"]
    tan_left_3d  = data["tan_left_3d"]
    rad_right_3d = data["rad_right_3d"]
    tan_right_3d = data["tan_right_3d"]

    fs = 512
    cut = int(3 * fs)
    rad_left_3d  = rad_left_3d[:, cut:, :]
    tan_left_3d  = tan_left_3d[:, cut:, :]
    rad_right_3d = rad_right_3d[:, cut:, :]
    tan_right_3d = tan_right_3d[:, cut:, :]

    # 数据拼接(先left后right) 
    rad_3d = np.vstack((rad_left_3d, rad_right_3d)) 
    tan_3d = np.vstack((tan_left_3d, tan_right_3d))

    # ---- Numba  ----
    print("Warming up Numba JIT...")

    _dummy = np.random.randn(512).astype(np.float64)
    _ = sampen_windowed_1d_numba(
        _dummy,
        m=2,
        r_factor=0.2,
        win_len=128,
        step=128
    )

    print("Numba JIT warm-up completed.\n")

    # ---- 计算样本熵 ----
    print("Computing rad_left SampEn...")
    rad_left_sampen = sampen_trial_channel_parallel(rad_left_3d)

    print("Computing tan_left SampEn...")
    tan_left_sampen = sampen_trial_channel_parallel(tan_left_3d)

    print("Computing rad_right SampEn...")
    rad_right_sampen = sampen_trial_channel_parallel(rad_right_3d)

    print("Computing tan_right SampEn...")
    tan_right_sampen = sampen_trial_channel_parallel(tan_right_3d)

    print("Computing rad SampEn...") 
    rad_sampen = sampen_trial_channel_parallel(rad_3d) 
    
    print("Computing tan SampEn...") 
    tan_sampen = sampen_trial_channel_parallel(tan_3d)

    # ---- 数值自检 ----
    inspect_sampen("rad_left",  rad_left_sampen)
    inspect_sampen("tan_left",  tan_left_sampen)
    inspect_sampen("rad_right", rad_right_sampen)
    inspect_sampen("tan_right", tan_right_sampen)
    inspect_sampen("rad", rad_sampen) 
    inspect_sampen("tan", tan_sampen)

    # ---- 保存结果 ----
    np.savez(
        result_file,
        rad_left_sampen   = rad_left_sampen,
        tan_left_sampen   = tan_left_sampen,
        rad_right_sampen  = rad_right_sampen,
        tan_right_sampen  = tan_right_sampen,
        rad_sampen        = rad_sampen, 
        tan_sampen        = tan_sampen,
        win_len    = 512,
        step       = 512,
        m          = 2,
        r_factor   = 0.2
    )

    print("\n样本熵计算、自检与保存全部完成。")

# =========================
# Windows 并行保护
# =========================
if __name__ == "__main__":
    main()

# %% 结果打印

SampEn = np.load("sampen_results.npz")

rad_Sampen = SampEn["rad_sampen"]
tan_Sampen = SampEn["tan_sampen"]

mean_rad_sampen = np.mean(rad_Sampen,axis=0)
mean_tan_sampen = np.mean(tan_Sampen,axis=0)
std_rad_sampen = np.std(rad_Sampen,axis=0)
std_tan_sampen = np.std(tan_Sampen,axis=0)

print('Sample Entropy\n','rad:\n','mean:', mean_rad_sampen, 'std:', std_rad_sampen,'\n'
                         'tan:\n','mean:', mean_tan_sampen, 'std:', std_tan_sampen)
# %%
