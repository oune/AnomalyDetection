from matplotlib.ft2font import BOLD
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sns

## normal / abnormal (sample rate : 4000 -> 17s)
# 20211015 48 / -
# 20220112 10 / 16
# 20220121 995 / 741
# 20220208 1691 / -
# 20220209 1273 / -
# 20220310 5158 / - 
# 20220311 1563 / -
# 20220315 1874 / -
# 20220323 - / 2129
# 20220331 - / 1460
# normal 전체 17m 소요, abnormal 전체 6m 소요 (raw 기준)

def load_data(path : str):
    data_raw = pd.read_csv(path, encoding='unicode_escape', delimiter='\t', header=None)
    
    if len(data_raw) >= 68000:
        data_raw_list = data_raw[1].values.tolist()
        data_raw_arr = np.array(data_raw_list)
        
        data_length = len(data_raw)
        index_remain = len(data_raw) % 68000
        data_list = data_raw_arr[:data_length-index_remain]

        data = data_list.reshape(int(data_length/68000), 68000)
        data = np.mean(np.abs(data), axis=1).reshape(-1,1)
    else:
        data = []

    return data

def load_data_raw(dir_path : str, is_normal : bool, is_train : bool, stop_idx : int):
    dir_list = os.listdir(dir_path)
    data_0_all, data_1_all, data_2_all, data_3_all = [], [], [], []

    if is_normal:
        file_name = "*_continuous.xlsx"
    else:
        file_name = "*_continuous_*fault.xlsx"

    for idx, dir in enumerate(tqdm(dir_list)):
        file_path_list = glob.glob(rf"{dir_path}\{dir}\{file_name}")
        
        if file_path_list != []:
            data_0_all.append(load_data(file_path_list[0]))
            data_1_all.append(load_data(file_path_list[1]))
            data_2_all.append(load_data(file_path_list[2]))
            data_3_all.append(load_data(file_path_list[3]))
        
        if stop_idx >= 0:
            if stop_idx == idx:
                break

    data_0_concate = concate_data(data_0_all, axis=0)
    data_1_concate = concate_data(data_1_all, axis=0)
    data_2_concate = concate_data(data_2_all, axis=0)
    data_3_concate = concate_data(data_3_all, axis=0)

    data_0, data_1, data_2, data_3 = [], [], [], []

    for elem_0, elem_1, elem_2, elem_3 in zip(data_0_concate, data_1_concate, data_2_concate, data_3_concate):
        if elem_0 >= 0.003 and elem_1 >= 0.003 and elem_2 >= 0.003 and elem_3 >= 0.003:
            data_0.append(elem_0)
            data_1.append(elem_1)
            data_2.append(elem_2)
            data_3.append(elem_3)

    data_0, data_1, data_2, data_3 = np.array(data_0), np.array(data_1), np.array(data_2), np.array(data_3)

    if is_train:
        split_rate = 0.9
        split_idx = int(len(data_0)*split_rate)

        train_data_0 = data_0[:split_idx]
        train_data_1 = data_1[:split_idx]
        train_data_2 = data_2[:split_idx]
        train_data_3 = data_3[:split_idx]
        validation_data_0 = data_0[split_idx:]
        validation_data_1 = data_1[split_idx:]
        validation_data_2 = data_2[split_idx:]
        validation_data_3 = data_3[split_idx:]

        return train_data_0, train_data_1, train_data_2, train_data_3, validation_data_0, validation_data_1, validation_data_2, validation_data_3
    else:
        return data_0, data_1, data_2, data_3

def data_split_to_chunk(dir_path : str, is_normal : bool, save_dir_path : str):
    os.makedirs(save_dir_path, exist_ok=True)

    dir_list = os.listdir(dir_path)
    data_0_all, data_1_all, data_2_all, data_3_all = [], [], [], []

    if is_normal:
        file_name = "*_continuous.xlsx"
    else:
        file_name = "*_continuous_*fault.xlsx"

    for idx, dir in enumerate(tqdm(dir_list)):
        file_path_list = glob.glob(rf"{dir_path}\{dir}\{file_name}")
        print(dir)

        if file_path_list != []:
            data_0_all.append(file_path_list[0])
            data_1_all.append(file_path_list[1])
            data_2_all.append(file_path_list[2])
            data_3_all.append(file_path_list[3])


    data_0_list, data_1_list, data_2_list, data_3_list = [], [], [], []

    for idx, (data_0, data_1, data_2, data_3) in enumerate(tqdm(zip(data_0_all, data_1_all, data_2_all, data_3_all))):
        data_0_raw = pd.read_csv(data_0, encoding='unicode_escape', delimiter='\t', header=None)
        data_1_raw = pd.read_csv(data_1, encoding='unicode_escape', delimiter='\t', header=None)
        data_2_raw = pd.read_csv(data_2, encoding='unicode_escape', delimiter='\t', header=None)
        data_3_raw = pd.read_csv(data_3, encoding='unicode_escape', delimiter='\t', header=None)

        if len(data_0_raw) >= 68000 and len(data_1_raw) >= 68000 and len(data_2_raw) >= 68000 and len(data_3_raw) >= 68000:
            data_0_raw_list = data_0_raw[1].values.tolist()
            data_1_raw_list = data_1_raw[1].values.tolist()
            data_2_raw_list = data_2_raw[1].values.tolist()
            data_3_raw_list = data_3_raw[1].values.tolist()
            data_0_raw_arr = np.array(data_0_raw_list)
            data_1_raw_arr = np.array(data_1_raw_list)
            data_2_raw_arr = np.array(data_2_raw_list)
            data_3_raw_arr = np.array(data_3_raw_list)

            data_length = len(data_0_raw)
            index_remain = len(data_0_raw) % 68000

            data_0_list.append(data_0_raw_arr[:data_length-index_remain])
            data_1_list.append(data_1_raw_arr[:data_length-index_remain])
            data_2_list.append(data_2_raw_arr[:data_length-index_remain])
            data_3_list.append(data_3_raw_arr[:data_length-index_remain])

    for data_0, data_1, data_2, data_3 in tqdm(zip(data_0_list, data_1_list, data_2_list, data_3_list)):
        data_0_re = data_0.reshape(int(len(data_0)/68000), 68000)
        data_1_re = data_1.reshape(int(len(data_1)/68000), 68000)
        data_2_re = data_2.reshape(int(len(data_2)/68000), 68000)
        data_3_re = data_3.reshape(int(len(data_3)/68000), 68000)

        ## dynamic 
        if len(data_0_re) == 16:
            dir_name = 20220112
        elif len(data_0_re) == 741:
            dir_name = 20220121
        elif len(data_0_re) == 2129:
            dir_name = 20220323
        elif len(data_0_re) == 1460:
            dir_name = 20220331
        ##

        save_idx = 0
        for (chunk_0, chunk_1, chunk_2, chunk_3) in zip(data_0_re, data_1_re, data_2_re, data_3_re):
            chunk_0_check = np.mean(np.abs(chunk_0), axis=0).reshape(-1,1)
            chunk_1_check = np.mean(np.abs(chunk_1), axis=0).reshape(-1,1)
            chunk_2_check = np.mean(np.abs(chunk_2), axis=0).reshape(-1,1)
            chunk_3_check = np.mean(np.abs(chunk_3), axis=0).reshape(-1,1)

            if chunk_0_check >= 0.003 and chunk_1_check >= 0.003 and chunk_2_check >= 0.003 and chunk_3_check >= 0.003:
                chunk_0_df = pd.DataFrame(chunk_0)
                chunk_1_df = pd.DataFrame(chunk_1)
                chunk_2_df = pd.DataFrame(chunk_2)
                chunk_3_df = pd.DataFrame(chunk_3)

                chunk_0_df.to_csv(rf'{save_dir_path}\abnormal_0_{dir_name}_{save_idx}.csv', index=False)
                chunk_1_df.to_csv(rf'{save_dir_path}\abnormal_1_{dir_name}_{save_idx}.csv', index=False)
                chunk_2_df.to_csv(rf'{save_dir_path}\abnormal_2_{dir_name}_{save_idx}.csv', index=False)
                chunk_3_df.to_csv(rf'{save_dir_path}\abnormal_3_{dir_name}_{save_idx}.csv', index=False)

                save_idx += 1

def data_to_fft(data_0, data_1, data_2, data_3):
    data_fft_0 = np.fft.fft(data_0) / len(data_0)
    data_fft_1 = np.fft.fft(data_1) / len(data_1)
    data_fft_2 = np.fft.fft(data_2) / len(data_2)
    data_fft_3 = np.fft.fft(data_3) / len(data_3)

    return data_fft_0, data_fft_1, data_fft_2, data_fft_3

def display_data(train_data, validation_data, test_data):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(train_data, label='Normal', color='blue', animated = True, linewidth=1)
    ax.plot(validation_data, label='Normal_valiidation', color='violet', animated = True, linewidth=1)
    ax.plot(test_data, label='Abrnomal', color='red', animated = True, linewidth=1)
    plt.legend(loc='upper right')
    ax.set_title('Frequency Sensor ', fontsize=16)
    plt.show()

def data_adjust_scale(scaler, data, is_fit : bool):
    # sklearn.preprocessing -> StandardScaler, MinMaxScaler, RobustScaler
    scaler = scaler

    if is_fit:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)

    return data

def data_reshape_for_train(data_0, data_1, data_2, data_3):
    data = concate_data((data_0, data_1, data_2, data_3), axis=1)
    data = data.reshape(data.shape[0], 1, data.shape[1])

    return data

def concate_data(arr, axis):
    data = np.concatenate(arr, axis=axis)

    return data

def stack_data(arr, axis):
    data = np.stack(arr, axis=axis)

    return data

def check_data_to_sacler(train_data, test_data):
    x = pd.DataFrame({
        'x1': train_data.squeeze(-1),
        'x2': test_data.squeeze(-1),
        })

    scaler = RobustScaler()
    robust_df = scaler.fit_transform(x)
    robust_df = pd.DataFrame(robust_df, columns =['x1', 'x2'])
    
    scaler = StandardScaler()
    standard_df = scaler.fit_transform(x)
    standard_df = pd.DataFrame(standard_df, columns =['x1', 'x2'])
    
    scaler = MinMaxScaler()
    minmax_df = scaler.fit_transform(x)
    minmax_df = pd.DataFrame(minmax_df, columns =['x1', 'x2'])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 5))
    ax1.set_title('Before Scaling')
    
    sns.kdeplot(x['x1'], ax = ax1, color ='r')
    sns.kdeplot(x['x2'], ax = ax1, color ='b')
    ax2.set_title('After Robust Scaling')
    
    sns.kdeplot(robust_df['x1'], ax = ax2, color ='red')
    sns.kdeplot(robust_df['x2'], ax = ax2, color ='blue')
    ax3.set_title('After Standard Scaling')
    
    sns.kdeplot(standard_df['x1'], ax = ax3, color ='black')
    sns.kdeplot(standard_df['x2'], ax = ax3, color ='g')
    ax4.set_title('After Min-Max Scaling')
    
    sns.kdeplot(minmax_df['x1'], ax = ax4, color ='black')
    sns.kdeplot(minmax_df['x2'], ax = ax4, color ='g')
    plt.show()