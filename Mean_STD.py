# %%  libraries
import numpy as np

# %% 函数包
# 数据点平均值的计算
def channel_mean(data):

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("Input data must be 3D: (trial, sample, channel)")

    mean = np.nanmean(data, axis=1)

    return mean

# %%  data loading
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

#数据拼接(先left再right)
rad_3d = np.vstack((rad_left_3d, rad_right_3d))
tan_3d = np.vstack((tan_left_3d, tan_right_3d))

# %% 计算每个trial的平均值
rad_left_ch_mean  = channel_mean(rad_left_3d)
rad_right_ch_mean = channel_mean(rad_right_3d)
tan_left_ch_mean  = channel_mean(tan_left_3d)
tan_right_ch_mean = channel_mean(tan_right_3d)
rad_ch_mean = channel_mean(rad_3d)
tan_ch_mean = channel_mean(tan_3d)

print('rad_left_ch_mean', rad_left_ch_mean.shape)
print('tan_left_ch_mean', tan_left_ch_mean.shape)
print('rad_right_ch_mean', rad_right_ch_mean.shape)
print('tan_right_ch_mean', tan_right_ch_mean.shape)
print("rad_ch_mean", rad_ch_mean.shape)
print("tan_ch_mean", tan_ch_mean.shape)

# %% 减去均值
rad_left_demean = rad_left_3d - rad_left_ch_mean[:, np.newaxis, :]
tan_left_demean = tan_left_3d - tan_left_ch_mean[:, np.newaxis, :]
rad_right_demean = rad_right_3d - rad_right_ch_mean[:, np.newaxis, :]
tan_right_demean = tan_right_3d - tan_right_ch_mean[:, np.newaxis, :]
rad_demean = rad_3d - rad_ch_mean[:, np.newaxis, :]
tan_demean = tan_3d - tan_ch_mean[:, np.newaxis, :]

# %%  再做 sample 维度平均和std，构造一个 robust 幅值特征
rad_left_mean  = np.nanmean(rad_left_demean, axis=1)
tan_left_mean  = np.nanmean(tan_left_demean, axis=1)
rad_right_mean = np.nanmean(rad_right_demean, axis=1)
tan_right_mean = np.nanmean(tan_right_demean, axis=1)
rad_mean = np.nanmean(rad_demean, axis=1)
tan_mean = np.nanmean(tan_demean, axis=1)

rad_left_std  = np.nanstd(rad_left_demean, axis=1)
tan_left_std  = np.nanstd(tan_left_demean, axis=1)
rad_right_std = np.nanstd(rad_right_demean, axis=1)
tan_right_std = np.nanstd(tan_right_demean, axis=1)
rad_std = np.nanstd(rad_demean, axis=1)
tan_std = np.nanstd(tan_demean, axis=1)
# %% 对trial维度做平均
rad_left_mean2 = np.mean(rad_left_mean, axis=0)
tan_left_mean2 = np.mean(tan_left_mean, axis=0)
rad_right_mean2 = np.mean(rad_right_mean, axis=0)
tan_right_mean2 = np.mean(tan_right_mean, axis=0)
rad_mean2 = np.mean(rad_mean, axis=0)
tan_mean2 = np.mean(tan_mean, axis=0)

rad_left_meanstd = np.std(rad_left_mean, axis=0)
tan_left_meanstd = np.std(tan_left_mean, axis=0)
rad_right_meanstd = np.std(rad_right_mean, axis=0)
tan_right_meanstd = np.std(tan_right_mean, axis=0)
rad_meanstd = np.std(rad_mean, axis=0)
tan_meanstd = np.std(tan_mean, axis=0)

rad_left_stdmean = np.mean(rad_left_std, axis=0)
tan_left_stdmean = np.mean(tan_left_std, axis=0)
rad_right_stdmean = np.mean(rad_right_std, axis=0)
tan_right_stdmean = np.mean(tan_right_std, axis=0)
rad_stdmean = np.mean(rad_std, axis=0)
tan_stdmean = np.mean(tan_std, axis=0)

rad_left_std2 = np.std(rad_left_std, axis=0)
tan_left_std2 = np.std(tan_left_std, axis=0)
rad_right_std2 = np.std(rad_right_std, axis=0)
tan_right_std2 = np.std(tan_right_std, axis=0)
rad_std2 = np.std(rad_std, axis=0)
tan_std2 = np.std(tan_std, axis=0)

# %% 打印结果
print('rad_left','mean2', rad_left_mean2, 'meanstd', rad_left_meanstd, 'stdmean', rad_left_stdmean, 'std2', rad_left_std2)
print('tan_left','mean2', tan_left_mean2, 'meanstd', tan_left_meanstd, 'stdmean', tan_left_stdmean, 'std2', tan_left_std2)
print('rad_right','mean2', rad_right_mean2, 'meanstd', rad_right_meanstd, 'stdmean', rad_right_stdmean, 'std2', rad_right_std2)
print('tan_right','mean2', tan_right_mean2, 'meanstd', tan_right_meanstd, 'stdmean', tan_right_stdmean, 'std2', tan_right_std2)
print('rad','mean2', rad_mean2, 'meanstd', rad_meanstd, 'stdmean', rad_stdmean, 'std2', rad_std2)
print('tan','mean2', tan_mean2, 'meanstd', tan_meanstd, 'stdmean', tan_stdmean, 'std2', tan_std2)
# %%  保存Mean_std结果
np.savez(
    "Mean_Result.npz",
    rad_left_mean = rad_left_mean,
    tan_left_mean = tan_left_mean,
    rad_right_mean = rad_right_mean, 
    tan_right_mean = tan_right_mean,
    rad_mean = rad_mean,
    tan_mean = tan_mean,   
    description = "Mean result of trial_data_3d, shape(trial, channel)"
)

np.savez(
    "Std_Result.npz",
    rad_left_std = rad_left_std,
    tan_left_std = tan_left_std,
    rad_right_std = rad_right_std,
    tan_right_std = tan_right_std,
    rad_std = rad_std,
    tan_std = tan_std,
    description = "Std result of trial_data_3d, shape(trial, channel)"
)
# %%
