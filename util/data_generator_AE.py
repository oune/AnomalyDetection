from sklearn import pipeline
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import enum
import glob
import pandas as pd
import numpy as np
from util.data_management import concate_data, stack_data
import math
import random
from sklearn.preprocessing import Normalizer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline

class TrainMode(enum.Enum):
    TRAIN = 'Train'
    VALIDATION = 'Validation'
    TEST = 'Test'

class DataGenerator(Sequence):
    def __init__(self, dataset_path : str, train_mode : str, batch_size : int, split : tuple, is_cache : bool, is_normalize : bool, is_lstm : bool, is_adjust_fit : bool, pipeline = None, is_predict = None) -> None:
        super().__init__()
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dataset_path = dataset_path
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.split = split
        self.is_cache = is_cache
        self.is_normalize = is_normalize
        self.is_lstm = is_lstm
        self.is_adjust_fit = is_adjust_fit
        self.is_predict = is_predict
        self.cache = {}
        self.predict_labels = []

        if pipeline == None:
            if self.is_normalize:
                self.pipeline = Pipeline([('Scaler' , StandardScaler()), ('Normalizer' , Normalizer())])
            else:
                self.pipeline = Pipeline([('Scaler' , StandardScaler())])
        else:
            self.pipeline = pipeline
    
        data_paths_0 = sorted(glob.glob(dataset_path + '\\Normal\\*_0_*.csv'))
        data_paths_1 = sorted(glob.glob(dataset_path + '\\Normal\\*_1_*.csv'))
        data_paths_2 = sorted(glob.glob(dataset_path + '\\Normal\\*_2_*.csv'))
        data_paths_3 = sorted(glob.glob(dataset_path + '\\Normal\\*_3_*.csv'))

        train_idx = int(len(data_paths_0)*self.split[0])
        validation_idx = train_idx + int(int(len(data_paths_0)*self.split[1]))

        if self.train_mode == TrainMode.TRAIN.value:
            data_paths_0 = data_paths_0[:train_idx]
            data_paths_1 = data_paths_1[:train_idx]
            data_paths_2 = data_paths_2[:train_idx]
            data_paths_3 = data_paths_3[:train_idx]
        elif self.train_mode == TrainMode.VALIDATION.value:
            data_paths_0 = data_paths_0[train_idx:validation_idx]
            data_paths_1 = data_paths_1[train_idx:validation_idx]
            data_paths_2 = data_paths_2[train_idx:validation_idx]
            data_paths_3 = data_paths_3[train_idx:validation_idx]
        elif self.train_mode == TrainMode.TEST.value:
            data_paths_0 = data_paths_0[validation_idx:]
            data_paths_1 = data_paths_1[validation_idx:]
            data_paths_2 = data_paths_2[validation_idx:]
            data_paths_3 = data_paths_3[validation_idx:]

            data_paths_0 = [(data_path_0, [1.]) for data_path_0 in data_paths_0]
            data_paths_1 = [(data_path_1, [1.]) for data_path_1 in data_paths_1]
            data_paths_2 = [(data_path_2, [1.]) for data_path_2 in data_paths_2]
            data_paths_3 = [(data_path_3, [1.]) for data_path_3 in data_paths_3]

            abnormals_0 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_0_*.csv'))
            abnormals_1 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_1_*.csv'))
            abnormals_2 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_2_*.csv'))
            abnormals_3 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_3_*.csv'))

            abnormals_0 = [(abnormal_0, [0.]) for abnormal_0 in abnormals_0]
            abnormals_1 = [(abnormal_1, [0.]) for abnormal_1 in abnormals_1]
            abnormals_2 = [(abnormal_2, [0.]) for abnormal_2 in abnormals_2]
            abnormals_3 = [(abnormal_3, [0.]) for abnormal_3 in abnormals_3]

            data_paths_0.extend(abnormals_0)
            data_paths_1.extend(abnormals_1)
            data_paths_2.extend(abnormals_2)
            data_paths_3.extend(abnormals_3)
       
        self.data_paths = list(zip(data_paths_0, data_paths_1, data_paths_2, data_paths_3))
        
    def __len__(self):
        return math.ceil(len(self.data_paths) / self.batch_size)
    
    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]
        
        dataset_batch = self.data_paths[index*self.batch_size:(index+1)*self.batch_size]
        data_0_batch, data_1_batch, data_2_batch, data_3_batch = [], [], [], []
        
        if self.train_mode == TrainMode.TEST.value:
            for data_0, _, _, _ in dataset_batch:
                self.predict_labels.append(data_0[1])

            dataset_batch = [(data[0][0], data[1][0], data[2][0], data[3][0]) for data in dataset_batch]

        for data_batch in dataset_batch:
            data_0 = pd.read_csv(data_batch[0], encoding='unicode_escape', delimiter='\t', header=None)
            data_1 = pd.read_csv(data_batch[1], encoding='unicode_escape', delimiter='\t', header=None)
            data_2 = pd.read_csv(data_batch[2], encoding='unicode_escape', delimiter='\t', header=None)
            data_3 = pd.read_csv(data_batch[3], encoding='unicode_escape', delimiter='\t', header=None)

            data_0 = np.array(data_0.values.tolist()[1:])
            data_1 = np.array(data_1.values.tolist()[1:])
            data_2 = np.array(data_2.values.tolist()[1:])
            data_3 = np.array(data_3.values.tolist()[1:])

            data_0 = np.mean(np.abs(data_0), axis=0)
            data_1 = np.mean(np.abs(data_1), axis=0)
            data_2 = np.mean(np.abs(data_2), axis=0)
            data_3 = np.mean(np.abs(data_3), axis=0)
            
            data_0_batch.append(data_0)
            data_1_batch.append(data_1)
            data_2_batch.append(data_2)
            data_3_batch.append(data_3)
        
        data_0_batch = self.adjust_data(data_0_batch)
        data_1_batch = self.adjust_data(data_1_batch)
        data_2_batch = self.adjust_data(data_2_batch)
        data_3_batch = self.adjust_data(data_3_batch)

        data = concate_data([data_0_batch, data_1_batch, data_2_batch, data_3_batch], axis=1)

        if self.is_lstm:
            data = data.reshape(data.shape[0], 1, 4)

        if self.train_mode == TrainMode.TEST.value:
            self.cache[index] = data
            return data
        else:
            self.cache[index] = data, data
            return data, data

    def on_epoch_end(self):
        if self.is_predict == None:
            datas = list(self.cache.values())
            random.shuffle(datas)

            self.cache = dict(zip(self.cache.keys(), datas))

    def adjust_data(self, data):        
        if self.is_adjust_fit:
            data = self.pipeline.fit_transform(data)
            self.is_adjust_fit = False
        else:
            data = self.pipeline.transform(data)

        return abs(data)