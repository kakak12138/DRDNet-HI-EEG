import os
import mne
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="This filename.*does not conform to MNE naming conventions")
warnings.filterwarnings("ignore", message=".*interpolated.*")

RAW_DATA_FOLDER = r'./data/CCSHI'
EVENTS_FOLDER = r'./data/CCSHI'
SAVE_PATH = r'./preprocess/CCSHI'

SPECIFIC_BAD_CHANNELS = {
    'A05': {'T': ['CP2']},
    'A07': {'T': ['CP1']}
}

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def preprocess_subject(subject_id):
    print(f"Processing subject: {subject_id}")

    for session_label in ['T', 'E']:
        print(f"Processing session: {session_label}")

        bdf_file = f'{subject_id}{session_label}.bdf'
        bdf_file_path = os.path.join(RAW_DATA_FOLDER, bdf_file)
        event_file = f'{subject_id}{session_label}_Event.txt'
        event_file_path = os.path.join(EVENTS_FOLDER, event_file)

        if not os.path.exists(bdf_file_path) or not os.path.exists(event_file_path):
            print(f"File not found: {bdf_file} or {event_file}")
            continue

        try:
            raw = mne.io.read_raw_bdf(bdf_file_path, preload=True, verbose=False, exclude=[])
        except Exception as e:
            print(f"Error reading BDF: {e}")
            continue

        if "Status" in raw.ch_names:
            raw.drop_channels(["Status"])

        current_session_bads = []
        if subject_id in SPECIFIC_BAD_CHANNELS:
            current_session_bads = SPECIFIC_BAD_CHANNELS[subject_id].get(session_label, [])

        if current_session_bads:
            try:
                montage = mne.channels.make_standard_montage('standard_1005')
                raw.set_montage(montage, verbose=False)
            except Exception as e:
                print(f"Error setting montage: {e}")

            valid_bads = [ch for ch in current_session_bads if ch in raw.ch_names]

            if valid_bads:
                raw.info['bads'].extend(valid_bads)
                print(f"Interpolating bad channels: {valid_bads}")
                raw.interpolate_bads(reset_bads=True, verbose=False)
            else:
                print(f"Bad channels not in channel list: {current_session_bads}")

        raw.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50, verbose=False)

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
            print(f"Error reading event file: {e}")
            continue

        if len(events) == 0:
            print("No events found")
            continue

        sfreq = raw.info["sfreq"]
        tmin = 0
        tmax = 4 - 1 / sfreq
        event_id = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

        epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                            reject=None, baseline=None, preload=True,
                            verbose=False, on_missing='warn')

        epochs.resample(250, npad="auto", verbose=False)

        data = epochs.get_data(copy=True)
        labels = epochs.events[:, -1]

        data = data * 1e6

        data_save_path = os.path.join(SAVE_PATH, f'{subject_id}{session_label}_data.npy')
        labels_save_path = os.path.join(SAVE_PATH, f'{subject_id}{session_label}_label.npy')

        np.save(data_save_path, data)
        np.save(labels_save_path, labels)

        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")


if __name__ == "__main__":
    subject_list = ['A01', 'A03', 'A05', 'A06', 'A07', 'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18',
                    'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25']
    for subject in subject_list:
        preprocess_subject(subject)
    print("Preprocessing completed")