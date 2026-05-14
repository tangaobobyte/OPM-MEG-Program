# %%  运行环境与基础设置
import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 设置 matplotlib 全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

italic_font = FontProperties(
    family='Times New Roman',
    style='italic',
    size=16
)

# %%  函数包
# 带通滤波计算 
def bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

# 50Hz陷波滤波
def notch_filter(data, notch_freq, fs, Q=30, axis=0):
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data, axis=axis)

# 按标签集分割数据集
def split_by_tag(data_seg, tag):
    """
    data_seg: (sample_points, n_trials, n_channels)
    tag: (n_trials,) or (n_trials, 1)
    """
    tag = np.asarray(tag).squeeze()

    left_idx = (tag == 0)
    right_idx = (tag == 1)

    leftdata_seg = data_seg[:, left_idx, :]
    rightdata_seg = data_seg[:, right_idx, :]

    return leftdata_seg, rightdata_seg

# 按通道分割数据集
def split_by_channel(data, rad_idx, tan_idx):
    rad_data = data[:, rad_idx, :]
    tan_data = data[:, tan_idx, :]
    return rad_data, tan_data

# 重塑数据组
def reshape_to_trial(data_2d, sample_points):
    """
    """
    n_total, n_channel = data_2d.shape
    assert n_total % sample_points == 0, "数据长度不能整除 sample_points"

    n_trial = n_total // sample_points
    data_3d = data_2d.reshape(n_trial, sample_points, n_channel)

    return data_3d

# %% 导入数据集
Dataset_Dir = r'C:\Users\Tangaobo\Desktop\nature_python\MEG_analysis\data'

Data_List = [
    'Data_01.txt',
    'Data_02.txt',
    'Data_03.txt',
    'Data_04.txt',
    'Data_05.txt',
    'Data_06.txt',
    'Data_07.txt',
    'Data_08.txt',
]

# 1. 读取所有数据
data_list = []
for fname in Data_List:
    data_path = os.path.join(Dataset_Dir, fname)
    data = np.loadtxt(data_path)   # (samplepoint_i, channel)
    data_list.append(data)

# 2. 通道一致性检查
channel = data_list[0].shape[1]
assert all(d.shape[1] == channel for d in data_list), "Channel 数不一致"

# 3. 构造三维数组
max_len = max(d.shape[0] for d in data_list)
dataset = len(data_list)

data_3d = np.full((max_len, channel, dataset), np.nan)

for i, d in enumerate(data_list):
    data_3d[:d.shape[0], :, i] = d * 1e6 
print("Final data shape:", data_3d.shape)

# %%  数据预处理
# 设置滤波器的参数
lowcut = 0.1  # 带通滤波器低频截止频率
highcut = 30.0  # 带通滤波器高频截止频率
notch_freq = 50  #陷波滤波器频率
fs = 512  #采样频率
T = 9  
sample_points = fs*T
filtered_data = np.full_like(data_3d, np.nan)  #生成预数据集

for i in range(data_3d.shape[2]):
    x = data_3d[:, :, i]
    valid_len = np.min([np.sum(~np.isnan(x[:, ch])) for ch in range(x.shape[1])])
    x_notch = notch_filter(x[:valid_len, :], notch_freq, fs)
    x_bp    = bandpass_filter(x_notch, lowcut, highcut, fs)

    filtered_data[:valid_len, :, i] = x_bp


# %%  分割通道划分数据集
assert filtered_data.shape[1] == 8, "当前函数仅适用于 8 通道数据"
radial_idx = [0, 2, 4, 6]       # 径向通道
tangential_idx = [1, 3, 5, 7]   # 切向通道

rad_filterdata, tan_filterdata = split_by_channel(filtered_data, radial_idx, tangential_idx)
print('径向数据集形状', rad_filterdata.shape)  # (samplepoint, 4, dataset)
print('切向数据集形状',tan_filterdata.shape)   # (samplepoint, 4, dataset)

# %% 重组左右数据集为二维数据集
samplepoint, channel, dataset = data_3d.shape
rad_data_list = []
tan_data_list = []

for i in range(dataset):
    rad_data = rad_filterdata[:, :, i]
    tan_data  = tan_filterdata[:, :, i]

    # 找出有效行（第一个通道非 NaN 即可判断该行是否有效）
    valid_idx = ~np.isnan(rad_data[:, 0])

    # 提取有效数据
    rad_valid = rad_data[valid_idx, :]
    tan_valid = tan_data[valid_idx, :]

    rad_data_list.append(rad_valid)
    tan_data_list.append(tan_valid)

# 拼接为二维数组
rad_2data = np.vstack(rad_data_list)
tan_2data = np.vstack(tan_data_list)
print("2D data shape of rad:", rad_2data.shape)
print("2D data shape of tan:", tan_2data.shape)

# %%  标签集导入
Tag_List = [
    'Tag_01.txt',
    'Tag_02.txt',
    'Tag_03.txt',
    'Tag_04.txt',
    'Tag_05.txt',
    'Tag_06.txt',
    'Tag_07.txt',
    'Tag_08.txt',
]

trial_count = np.zeros(len(Tag_List), dtype=int)
tag_list = []

for i, fname in enumerate(Tag_List):
    file_path = os.path.join(Dataset_Dir, fname)
    tag = np.loadtxt(file_path)

    tag = np.asarray(tag).reshape(-1)   # 保证一维
    trial_count[i] = tag.shape[0]
    tag_list.append(tag)

tagset = np.concatenate(tag_list)
print(trial_count)

# %%  根据标签集划分数据集
assert rad_2data.shape[0]/sample_points == tan_2data.shape[0]/sample_points == tagset.shape[0]

rad_left_list = []
rad_right_list = []
tan_left_list = []
tan_right_list = []

n_trials = tagset.shape[0]

for k in range(n_trials):
    start = k * sample_points
    end   = (k + 1) * sample_points

    rad_trial = rad_2data[start:end, :]   # (sample_points, 4)
    tan_trial = tan_2data[start:end, :]   # (sample_points, 4)

    if tagset[k] == 0:
        rad_left_list.append(rad_trial)
        tan_left_list.append(tan_trial)
    elif tagset[k] == 1:
        rad_right_list.append(rad_trial)
        tan_right_list.append(tan_trial)
    else:
        raise ValueError(f"Unexpected tag value {tagset[k]} at trial {k}")

# 重新拼接为二维数组
rad_left_data  = np.vstack(rad_left_list)   if rad_left_list  else np.empty((0, rad_2data.shape[1]))
rad_right_data = np.vstack(rad_right_list)  if rad_right_list else np.empty((0, rad_2data.shape[1]))
tan_left_data  = np.vstack(tan_left_list)   if tan_left_list  else np.empty((0, tan_2data.shape[1]))
tan_right_data = np.vstack(tan_right_list)  if tan_right_list else np.empty((0, tan_2data.shape[1]))

print("rad_left_data shape :", rad_left_data.shape)
print("rad_right_data shape:", rad_right_data.shape)
print("tan_left_data shape :", tan_left_data.shape)
print("tan_right_data shape:", tan_right_data.shape)

#重塑数组，将 (trial*sample_points, channel) 重塑为(trial, sample_points, channel)
rad_left_3d  = reshape_to_trial(rad_left_data,  sample_points)
tan_left_3d  = reshape_to_trial(tan_left_data,  sample_points)

rad_right_3d = reshape_to_trial(rad_right_data, sample_points)
tan_right_3d = reshape_to_trial(tan_right_data, sample_points)

# %% 保存数据，方便调用
np.savez(
    "trial_data_3d.npz",
    rad_left_3d   = rad_left_3d,
    tan_left_3d   = tan_left_3d,
    rad_right_3d  = rad_right_3d,
    tan_right_3d  = tan_right_3d,
    fs            = fs,
    sample_points = 4608,
    channels      = ["C1", "C2", "C3", "C4"],
    description   = "Rad/Tan, Left/Right, trial-level 3D data"
)

np.savez(
    "tagset.npz",
    tagset = tagset,
    description     = "1D concatenated Tag_List: (trial)"
)
# %%
