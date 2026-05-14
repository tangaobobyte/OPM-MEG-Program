# =========================
# 对平均值、方差、信息熵、三阶矩、四阶矩分析结果进行配对t检验
# =========================
# %%  配置环境
import numpy as np
from scipy import stats

# %% 函数包
def paired_ttest_trial_then_channel(data1, data2, alpha=0.05):

    # -------- 基本检查 --------
    if data1.shape != data2.shape:
        raise ValueError("data1 和 data2 的形状必须一致")

    if data1.ndim != 2:
        raise ValueError("输入数据必须是二维数组 [trial, channel]")

    n_trial, n_channel = data1.shape

    print("=" * 70)
    print(f"Paired t-test: trial-wise within each channel")
    print(f"trial = {n_trial}, channel = {n_channel}")
    print("=" * 70)

    # -------- 每个 channel 的 trial-wise 配对 t 检验 --------
    t_stats  = np.zeros(n_channel)
    p_values = np.zeros(n_channel)
    cohens_d = np.zeros(n_channel)

    print("\n[1] Channel-wise paired t-test (trial level)")
    print("-" * 70)

    for ch in range(n_channel):
        x = data1[:, ch]
        y = data2[:, ch]

        t, p = stats.ttest_rel(x, y, nan_policy='omit')

        diff = x - y
        d = diff.mean() / diff.std(ddof=1)

        t_stats[ch]  = t
        p_values[ch] = p
        cohens_d[ch] = d

        print(
            f"Channel {ch:02d} | "
            f"t = {t: .4f}, "
            f"p = {p: .4f}, "
            f"Cohen's d = {d: .4f}"
        )

    # -------- channel 维度的统计汇总（平均）--------
    print("\n[2] Channel-level summary (mean over channels)")
    print("-" * 70)

    print(f"Mean t statistic : {t_stats.mean(): .4f}")
    print(f"Mean p value     : {p_values.mean(): .4f}  (descriptive only)")
    print(f"Mean Cohen's d   : {cohens_d.mean(): .4f}")

    sig_num = np.sum(p_values < alpha)
    print(f"Significant channels (p < {alpha}): {sig_num} / {n_channel}")

    print("=" * 70)

    results = {
        "channel": {
            "t_stat": t_stats,
            "p_value": p_values,
            "cohens_d": cohens_d
        },
        "channel_mean": {
            "t_stat": t_stats.mean(),
            "p_value": p_values.mean(),
            "cohens_d": cohens_d.mean()
        }
    }

    return results

# %% 平均值配对t检验
# 导入平均值结果
Mean_result_file = "Mean_Result.npz"
Mean_res = np.load(Mean_result_file)

rad_left_mean = Mean_res["rad_left_mean"]
tan_left_mean = Mean_res["tan_left_mean"]
rad_right_mean = Mean_res["rad_right_mean"]
tan_right_mean = Mean_res["tan_right_mean"]

# 结果拼接(先left后right)
rad_mean = np.vstack((rad_left_mean, rad_right_mean))
tan_mean = np.vstack((tan_left_mean, tan_right_mean))

# t检验
Mean_left_ttest  = paired_ttest_trial_then_channel(rad_left_mean, tan_left_mean)
Mean_right_ttest = paired_ttest_trial_then_channel(rad_right_mean, tan_right_mean)
Mean_both_ttest  = paired_ttest_trial_then_channel(rad_mean, tan_mean)

# %% Std配对t检验
# 导入std结果
Std_result_file = "Std_Result.npz"
Std_res = np.load(Std_result_file)

rad_left_std = Std_res["rad_left_std"]
tan_left_std = Std_res["tan_left_std"]
rad_right_std = Std_res["rad_right_std"]
tan_right_std = Std_res["tan_right_std"]

# 结果拼接(先left后right)
rad_std = np.vstack((rad_left_std, rad_right_std))
tan_std = np.vstack((tan_left_std, tan_right_std))

# t检验
Std_left_ttest  = paired_ttest_trial_then_channel(rad_left_std, tan_left_std)
Std_right_ttest = paired_ttest_trial_then_channel(rad_right_std, tan_right_std)
Std_both_ttest  = paired_ttest_trial_then_channel(rad_std, tan_std)

# %% 样本熵t检验
# 导入样本熵结果
Entropy_result_file = "sampen_results.npz"
entropy_res = np.load(Entropy_result_file)

rad_left_sampen  = entropy_res["rad_left_sampen"]
tan_left_sampen  = entropy_res["tan_left_sampen"]
rad_right_sampen = entropy_res["rad_right_sampen"]
tan_right_sampen = entropy_res["tan_right_sampen"]

# 结果拼接(先left后right)
rad_sampen = np.vstack((rad_left_sampen, rad_right_sampen))
tan_sampen = np.vstack((tan_left_sampen, tan_right_sampen))

# t检验
Entropy_left_ttest = paired_ttest_trial_then_channel(rad_left_sampen, tan_left_sampen)
Entropy_right_ttest = paired_ttest_trial_then_channel(rad_right_sampen, tan_right_sampen)
Entropy_both_ttest = paired_ttest_trial_then_channel(rad_sampen, tan_sampen)

# %% 三阶矩偏度skewness配对t检验
# 导入skewness结果
Stew_result_file = "skewness_result.npz"
Stew_res = np.load(Stew_result_file)

rad_left_stew = Stew_res["skewness_rad_left"]
tan_left_stew = Stew_res["skewness_tan_left"]
rad_right_stew = Stew_res["skewness_rad_right"]
tan_right_stew = Stew_res["skewness_tan_right"]

# 结果拼接(先left后right)
rad_stew = np.vstack((rad_left_stew,rad_right_stew))
tan_stew = np.vstack((tan_left_stew, tan_right_stew))

# t检验
stew_left_ttest = paired_ttest_trial_then_channel(rad_left_stew, tan_left_stew)
stew_right_ttest = paired_ttest_trial_then_channel(rad_right_stew, tan_right_stew)
stew_both_ttest = paired_ttest_trial_then_channel(rad_stew, tan_stew)

# %% 四阶矩峰度kurtosis配对t检验
# 导入kurtosis结果
Kurt_result_file = "kurtosis_result.npz"
Kurt_res = np.load(Kurt_result_file)

rad_left_kurt = Kurt_res["kurtosis_rad_left"]
tan_left_kurt = Kurt_res["kurtosis_tan_left"]
rad_right_kurt = Kurt_res["kurtosis_rad_right"]
tan_right_kurt = Kurt_res["kurtosis_tan_right"]

# 结果拼接(先left后right)
rad_kurt = np.vstack((rad_left_kurt, rad_right_kurt))
tan_kurt = np.vstack((tan_left_kurt, tan_right_kurt))

# t检验
kurt_left_ttest = paired_ttest_trial_then_channel(rad_left_kurt, tan_left_kurt)
kurt_right_ttest = paired_ttest_trial_then_channel(rad_right_kurt, tan_right_kurt)
kurt_both_ttest = paired_ttest_trial_then_channel(rad_kurt, tan_kurt)
# %%
