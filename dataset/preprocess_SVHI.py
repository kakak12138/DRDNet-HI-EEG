import os
import mne
import numpy as np
import warnings

# 忽略 MNE 的非标准命名警告
warnings.filterwarnings("ignore", message="This filename.*does not conform to MNE naming conventions")
warnings.filterwarnings("ignore", message=".*interpolated.*")

# --- 配置路径 ---
RAW_DATA_FOLDER = r'./data/SVHI'
EVENTS_FOLDER = r'./data/SVHI'
SAVE_PATH = r'./preprocess/SVHI'


SPECIFIC_BAD_CHANNELS = {
    'A05': {'T': ['CP2'], 'E': ['CP1']},
    'A07': {'T': ['CP1']},
    'A19': {'E': ['CP2']}
}

# 确保保存目录存在
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def preprocess_subject(subject_id):
    print(f"\n--- 正在处理被试: {subject_id} ---")

    for session_label in ['T', 'E']:
        print(f"  > 处理 {session_label} 会话...")

        bdf_file = f'{subject_id}{session_label}.bdf'
        bdf_file_path = os.path.join(RAW_DATA_FOLDER, bdf_file)
        event_file = f'{subject_id}{session_label}_Event.txt'
        event_file_path = os.path.join(EVENTS_FOLDER, event_file)

        if not os.path.exists(bdf_file_path) or not os.path.exists(event_file_path):
            print(f"    [跳过] 未找到文件: {bdf_file} 或 {event_file}")
            continue

        # --- 步骤 1: 加载数据 ---
        try:
            # exclude=[] 确保读取所有通道
            raw = mne.io.read_raw_bdf(bdf_file_path, preload=True, verbose=False, exclude=[])
        except Exception as e:
            print(f"    [错误] 读取 BDF 失败: {e}")
            continue

        if "Status" in raw.ch_names:
            raw.drop_channels(["Status"])

        # --- 处理坏道与插值 ---
        current_session_bads = []
        if subject_id in SPECIFIC_BAD_CHANNELS:
            current_session_bads = SPECIFIC_BAD_CHANNELS[subject_id].get(session_label, [])

        if current_session_bads:
            try:
                # [修改点] 根据元数据 "10-10 system"，使用 standard_1005 更精准
                montage = mne.channels.make_standard_montage('standard_1005')
                raw.set_montage(montage, verbose=False)
            except Exception as e:
                print(f"    [警告] 设置 Montage 失败: {e}")

            valid_bads = [ch for ch in current_session_bads if ch in raw.ch_names]

            if valid_bads:
                raw.info['bads'].extend(valid_bads)
                print(f"    [处理] 发现坏道 ({session_label}): {valid_bads} -> 执行插值")
                raw.interpolate_bads(reset_bads=True, verbose=False)
            else:
                print(f"    [提示] 预设坏道 {current_session_bads} 不在通道列表中，跳过。")

        # [确认] 元数据 Reference 为 CPz，用户要求不进行重参考 (Keep Original Reference)

        # --- 步骤 2: 滤波 (元数据 PowerLineFrequency: 50Hz) ---
        raw.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, verbose=False)

        # --- 步骤 3: 事件处理 ---
        events = []
        try:
            with open(event_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        sample, event_type = int(parts[0]), int(parts[1])
                        events.append([sample, 0, event_type])
            events = np.array(events)
            if len(events) > 0:
                events = events[events[:, 0].argsort()]
        except Exception as e:
            print(f"    [错误] 读取事件文件失败: {e}")
            continue

        if len(events) == 0:
            print("    [警告] 未找到任何事件，跳过。")
            continue

        # --- 步骤 4: 切分 Epochs ---
        sfreq = raw.info["sfreq"]
        tmin = 0
        tmax = 4 - 1 / sfreq
        event_id = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                            reject=None, baseline=None, preload=True,
                            verbose=False, on_missing='warn')

        # 重采样 (1000Hz -> 250Hz)
        epochs.resample(250, npad="auto", verbose=False)

        data = epochs.get_data(copy=True)
        labels = epochs.events[:, -1]

        # 单位转换
        data = data * 1e6

        # --- 步骤 5: 保存 ---
        data_save_path = os.path.join(SAVE_PATH, f'{subject_id}{session_label}_data.npy')
        labels_save_path = os.path.join(SAVE_PATH, f'{subject_id}{session_label}_label.npy')

        np.save(data_save_path, data)
        np.save(labels_save_path, labels)

        print(f"    [完成] Data: {data.shape}, Labels: {labels.shape}")


if __name__ == "__main__":
    subject_list = ['A01', 'A03', 'A05', 'A06', 'A07', 'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18',
                    'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25']
    for subject in subject_list:
        preprocess_subject(subject)
    print("\n" + "=" * 30 + "\n   所有预处理任务完成\n" + "=" * 30)