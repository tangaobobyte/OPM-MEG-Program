# %%  运行环境与基础设置
import numpy as np

# %%  函数包
#  计算三维MEG数据的三阶与四阶矩
def third_fourth_moment(data_3d, standardized=True):

    # 沿 time 维度去均值
    mean = data_3d.mean(axis=1, keepdims=True)
    Xc = data_3d - mean

    # 三阶、四阶中心矩（time 维度）
    m3 = np.mean(Xc ** 3, axis=1)
    m4 = np.mean(Xc ** 4, axis=1)

    if standardized:
        std = Xc.std(axis=1, ddof=1)
        skewness = m3 / (std ** 3)
        kurtosis = m4 / (std ** 4) - 3
        return skewness, kurtosis
    else:
        return m3, m4

# %% 加载数据
data = np.load('trial_data_3d.npz')

rad_left_3d  = data['rad_left_3d']
tan_left_3d  = data['tan_left_3d']
rad_right_3d = data['rad_right_3d']
tan_right_3d = data['tan_right_3d']

fs = 512
cut = int(3 * fs)

rad_left_3d  = rad_left_3d[:, cut:, :]
tan_left_3d  = tan_left_3d[:, cut:, :]
rad_right_3d = rad_right_3d[:, cut:, :]
tan_right_3d = tan_right_3d[:, cut:, :]

#数据拼接(先left再right)
rad_3d = np.vstack((rad_left_3d, rad_right_3d))
tan_3d = np.vstack((tan_left_3d, tan_right_3d))

# %% 计算（推荐使用标准化统计量）
results = {}

results['rad_left']  = third_fourth_moment(rad_left_3d)
results['tan_left']  = third_fourth_moment(tan_left_3d)
results['rad_right'] = third_fourth_moment(rad_right_3d)
results['tan_right'] = third_fourth_moment(tan_right_3d)
results['rad']       = third_fourth_moment(rad_3d)
results['tan']       = third_fourth_moment(tan_3d)

skewness_rad_left, kurtosis_rad_left = results['rad_left']
skewness_tan_left, kurtosis_tan_left = results['tan_left']
skewness_rad_right, kurtosis_rad_right = results['rad_right']
skewness_tan_right, kurtosis_tan_right = results['tan_right']
skewness_rad, kurtosis_rad = results['rad']
skewness_tan, kurtosis_tan = results['tan']

# %% 计算三阶矩、四阶矩的平均值和std
mean_skew_rad = np.mean(skewness_rad, axis=0)
std_skew_rad  = np.std(skewness_rad, axis=0)
mean_skew_tan = np.mean(skewness_tan, axis=0)
std_skew_tan  = np.std(skewness_tan, axis= 0)

mean_kurt_rad = np.mean(kurtosis_rad, axis=0)
std_kurt_rad  = np.std(kurtosis_rad, axis=0)
mean_kurt_tan = np.mean(kurtosis_tan, axis=0)
std_kurt_tan  = np.std(kurtosis_tan, axis=0)

# %% 打印结果
print('Skewness\n','mean','rad:',mean_skew_rad,'tan:',mean_skew_tan,'\n',
                   'std','rad:', std_skew_rad, 'tan:', std_skew_tan)

print('Kurtosis\n','mean','rad:',mean_kurt_rad,'tan:',mean_kurt_tan,'\n',
                   'std','rad:', std_kurt_rad, 'tan:', std_kurt_tan)


# %% 数据保存
np.savez(
    "skewness_result.npz",
    skewness_rad_left = skewness_rad_left,
    skewness_tan_left = skewness_tan_left,
    skewness_rad_right = skewness_rad_right,
    skewness_tan_right = skewness_tan_right,
    skewness_rad = skewness_rad,
    skewness_tan = skewness_tan,
    description = "third_moment of MEG, shape(trial, channel)"
)

np.savez(
    "kurtosis_result.npz",
    kurtosis_rad_left = kurtosis_rad_left,
    kurtosis_tan_left = kurtosis_tan_left,
    kurtosis_rad_right = kurtosis_rad_right,
    kurtosis_tan_right = kurtosis_tan_right,
    kurtosis_rad = kurtosis_rad,
    kurtosis_tan = kurtosis_tan,
    description = "fourth_moment of MEG, shape(trial, channel)"
)
# %%
