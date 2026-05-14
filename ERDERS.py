# %%  运行环境与基础设置
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

italic_font = FontProperties(
    family='Times New Roman',
    style='italic',
    size=16
)

# %% 

def compute_erd_timecourse(data_3d, baseline_idx):
    """
    data_3d: [trial, sample_points, channel]
    return : [trial, sample_points, channel]
    """
    # instantaneous power
    power = data_3d ** 2

    # baseline mean power (over time)
    P_baseline = np.mean(
        power[:, baseline_idx, :],
        axis=1,
        keepdims=True
    )  # [trial, 1, channel]

    erd = (power - P_baseline) / P_baseline
    return erd


def moving_average_time(data, win):

    kernel = np.ones(win) / win
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=0,
        arr=data
    )

def plot_erd_left_right_4ch(
    erd_left,
    erd_right,
    fs,
    ch_names=("C3", "FC3", "C4", "FC4")
):
    """
    erd_left[ch]  : [time, 2] (rad, tan)
    erd_right[ch] : [time, 2] (rad, tan)
    """
    n_time = next(iter(erd_left.values())).shape[0]
    t = np.arange(n_time) / fs

    plt.figure(figsize=(12, 8))

    for i, ch in enumerate(ch_names):
        plt.subplot(2, 2, i + 1)

        # ---- Left Hand ----
        plt.plot(
            t,
            erd_left[ch][:, 0] * 100,
            label="Left Radial",
            linewidth=2
        )
        plt.plot(
            t,
            erd_left[ch][:, 1] * 100,
            linestyle="--",
            label="Left Tangential",
            linewidth=2
        )

        # ---- Right Hand ----
        plt.plot(
            t,
            erd_right[ch][:, 0] * 100,
            label="Right Radial",
            linewidth=2,
            alpha=0.8
        )
        plt.plot(
            t,
            erd_right[ch][:, 1] * 100,
            linestyle="--",
            label="Right Tangential",
            linewidth=2,
            alpha=0.8
        )

        plt.axhline(0, linestyle=":", linewidth=1)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.title(ch, fontsize = 16)
        plt.xlim(-0.2, 9)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("Time (s)", fontsize = 16)
        plt.ylabel("ERD / ERS (%)", fontsize = 16)

        # 只在第一个子图显示 legend
        if i == 0:
            plt.legend(
                loc="upper left",
                bbox_to_anchor=(0.1, 0.98),
                fontsize=12,
                frameon=False
            )


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
    "ERDERS.pdf",
    format="pdf",
    bbox_inches="tight"
    )
    plt.show()

# %%  data loading
data = np.load("trial_data_3d.npz")

rad_left_3d  = data["rad_left_3d"]
tan_left_3d  = data["tan_left_3d"]
rad_right_3d = data["rad_right_3d"]
tan_right_3d = data["tan_right_3d"]

fs = 512  

baseline_idx = slice(0, 2 * fs)     # [0 : 2*fs]

idx_central = [0, 2]   # C3, C4
idx_frontocentral = [1, 3]  # FC3, FC4

# %% 
erd_rad_left  = compute_erd_timecourse(rad_left_3d, baseline_idx)
erd_tan_left  = compute_erd_timecourse(tan_left_3d, baseline_idx)

erd_rad_right = compute_erd_timecourse(rad_right_3d, baseline_idx)
erd_tan_right = compute_erd_timecourse(tan_right_3d, baseline_idx)

# %%  
erd_rad_left_mean = np.mean(erd_rad_left, axis=0)
erd_tan_left_mean = np.mean(erd_tan_left, axis=0)

erd_rad_right_mean = np.mean(erd_rad_right, axis=0)
erd_tan_right_mean = np.mean(erd_tan_right, axis=0)

# %% 
win = int(0.1 * fs)  # 200 ms 滑窗

erd_rad_left_smooth  = moving_average_time(erd_rad_left_mean,  win)
erd_tan_left_smooth  = moving_average_time(erd_tan_left_mean,  win)

erd_rad_right_smooth = moving_average_time(erd_rad_right_mean, win)
erd_tan_right_smooth = moving_average_time(erd_tan_right_mean, win)

# %%  
erd_central_left = np.stack(
    [
        erd_rad_left_smooth[:, idx_central],
        erd_tan_left_smooth[:, idx_central]
    ],
    axis=-1
)
# shape: [time, 2 (C3,C4), 2 (rad,tan)]
erd_frontocentral_left = np.stack(
    [
        erd_rad_left_smooth[:, idx_frontocentral],
        erd_tan_left_smooth[:, idx_frontocentral]
    ],
    axis=-1
)
# shape: [time, 2 (FC3,FC4), 2 (rad,tan)]
erd_central_right = np.stack(
    [
        erd_rad_right_smooth[:, idx_central],
        erd_tan_right_smooth[:, idx_central]
    ],
    axis=-1
)
# shape: [time, 2 (C3,C4), 2 (rad,tan)]
erd_frontocentral_right = np.stack(
    [
        erd_rad_right_smooth[:, idx_frontocentral],
        erd_tan_right_smooth[:, idx_frontocentral]
    ],
    axis=-1
)
# shape: [time, 2 (FC3,FC4), 2 (rad,tan)]

# %%  
erd_left = {
    "C3":  erd_central_left[:, 0, :],
    "C4":  erd_central_left[:, 1, :],
    "FC3": erd_frontocentral_left[:, 0, :],
    "FC4": erd_frontocentral_left[:, 1, :]
}

erd_right = {
    "C3":  erd_central_right[:, 0, :],
    "C4":  erd_central_right[:, 1, :],
    "FC3": erd_frontocentral_right[:, 0, :],
    "FC4": erd_frontocentral_right[:, 1, :]
}


plot_erd_left_right_4ch(
    erd_left,
    erd_right,
    fs=512
)

# %%
