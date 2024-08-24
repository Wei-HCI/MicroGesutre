import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.skeleton import *

def load_data(data_path, num_frame, use_quaternion=True):
    data_list = []
    label_list = []
    state_list = []
    null_state = 4  # state label for static gestures
    default_label = 0  # Assigning a default label for all data points

    for file in tqdm(sorted(glob.glob(data_path))):
        try:
            csv_data = read_data_from_csv(file, use_quaternion)
            data_len, col = csv_data.shape
            #print(f"Original data length: {data_len}, Columns: {col}, csv:{file}")

            if data_len < num_frame:
                print(f"Data length ({data_len}) <= num_frame ({num_frame}). Exiting.")
                return [], [], []

            if data_len % num_frame != 0:
                discard_frame = data_len % num_frame
                csv_data = csv_data[discard_frame:]
                #print(f"Data length after discarding {discard_frame} frames: {csv_data.shape[0]}")

            csv_data = np.split(csv_data, len(csv_data) // num_frame, axis=0)
            #print(f"Number of data splits: {len(csv_data)}")

            for data_point in csv_data:
                if 'fist' in file:
                    label_list.append(0)
                elif 'cube' in file:
                    label_list.append(1)
                elif 'vertical' in file:
                    label_list.append(2)
                elif 'ring' in file:
                    label_list.append(3)
                elif 'pinky' in file:
                    label_list.append(4)
                elif 'scissors' in file:
                    label_list.append(5)
                elif 'slide' in file:
                    label_list.append(6)
                elif 'null' in file:
                    label_list.append(7)
                else:
                    continue

                if 'state' in data_point.columns:
                    state_list.append(data_point['state'].to_numpy())
                    data_point = data_point.drop('state', axis=1)
                else:
                    state_list.append(null_state * np.ones(data_point.shape[0]))

                data_point = np.array(np.split(data_point, 11, axis=1))
               # print(f"Data point shape after splitting: {data_point.shape}")
                data_point = np.swapaxes(data_point, 0, 1)
                data_list.append(data_point)

            if len(data_list) == 0:
                #print("No data points to stack.")
                return [], [], []

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    data_array = np.stack(data_list, axis=0)
    label_array = np.array(label_list)
    state_array = np.stack(state_list, axis=0)
    # print(f"Label data type: {label_array.dtype}")
    # print(f"Minimum label: {min(label_list)}, Maximum label: {max(label_list)}")
    # print(f"Data shape: {data_array.shape}")
    # print(f"Label shape: {label_array.shape}")
    # print(f"State shape: {state_array.shape}")

    return data_array, label_array, state_array

def read_data_from_csv(data_path, use_quaternion=True):
    gesture_clip = pd.read_csv(data_path)
    gesture_clip = gesture_clip.iloc[:, 8:]  # Adjust this based on actual data structure
    return gesture_clip

def change_state(data_path):
    gesture_clip = pd.read_csv(data_path)
    transition_1 = 0
    transition_2 = 0

    if "cut" in data_path or "spray" in data_path:
        for header in list(gesture_clip.columns):
            if 'state' in header:
                state_column = gesture_clip[header].to_numpy()

                for i in range(state_column.shape[0] - 1):
                    if state_column[i] == 1 and state_column[i+1] == 0:
                        transition_1 = i
                    elif state_column[i] == 0 and state_column[i+1] == 1:
                        transition_2 = i

                first = transition_1//2
                second = (transition_2 - transition_1)//4
                third = transition_2 - second
                fourth = (state_column.shape[0] - transition_2) // 2

                state_column[first:transition_1] = -1
                state_column[transition_1:(transition_1 + second)] = -2
                state_column[third:transition_2] = -2
                state_column[transition_2:(transition_2+fourth)] = -1

                for i in range(state_column.shape[0]):
                    if state_column[i] == 1:
                        state_column[i] = 3
                    elif state_column[i] == -1:
                        state_column[i] = 2
                    elif state_column[i] == -2:
                        state_column[i] = 1
    else:
        for header in list(gesture_clip.columns):
            if 'state' in header:
                gesture_clip[header] = 4

    new_path = data_path.replace('training_data', 'training_data_multistate')
    new_dir = new_path.split("gesture_")[0]
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    gesture_clip.to_csv(new_path, index=False)

def selected_frame(data, num_frame):
    frame, dim = data.shape
    if frame == num_frame:
        return data
    interval = frame / num_frame
    uniform_list = [int(i * interval) for i in range(num_frame)]
    return data.iloc[uniform_list]
