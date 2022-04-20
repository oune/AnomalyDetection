from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob

def split_data(raw_array):
    list = []

    begin = 0
    end = 68000

    while(1):
        list.append(raw_array[begin:end])
        begin += 68000
        end += 68000

        if end >= len(raw_array):
            list.append(raw_array[begin:])
            break
        
    return list

def load_data(path : str):
    data = np.array(pd.read_csv(path, encoding='unicode_escape', delimiter='\t', header=None, usecols=[1]))
    data = split_data(data)
    data = data[:len(data) - 1]

    return data

def load_data_AE(path : str):
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

def make_dataset():
    dir_list = os.listdir(r"C:\Users\VIP444\Documents\Anomaly-Dataset\sar400_vibration_data")
    data_0_all = []
    data_1_all = []
    data_2_all = []
    data_3_all = []

    for idx, dir in enumerate(tqdm(dir_list)):
        file_path_list = glob.glob(rf"C:\Users\VIP444\Documents\Anomaly-Dataset\sar400_vibration_data\{dir}\*_continuous.xlsx")

        data_0_all.append(load_data_AE(file_path_list[0]))
        data_1_all.append(load_data_AE(file_path_list[1]))
        data_2_all.append(load_data_AE(file_path_list[2]))
        data_3_all.append(load_data_AE(file_path_list[3]))

def concate_data(arr, axis):
    data = np.concatenate(arr, axis=axis)

    return data

def stack_data(arr, axis):
    data = np.stack(arr, axis=axis)

    return data