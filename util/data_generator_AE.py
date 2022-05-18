from sklearn import pipeline
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import enum
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from util.data_management import concate_data, stack_data

class TrainMode(enum.Enum):
    TRAIN = 'Train'
    VALIDATION = 'Validation'
    TEST = 'Test'

class DataGenerator(Sequence):
    def __init__(self, dataset_path : str, train_mode : str, batch_size : int, split : tuple, is_train : bool, is_cache : bool, is_normalize : bool, is_lstm : bool) -> None:
        super().__init__()
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dataset_path = dataset_path
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.split = split
        self.is_train = is_train
        self.is_cache = is_cache
        self.is_normalize = is_normalize
        self.is_lstm = is_lstm
        self.is_adjust_fit = True
        self.cache = {}

        if self.is_normalize:
            self.pipeline = Pipeline([('Normalizer' , Normalizer()), ('Scaler' , MinMaxScaler())])
        else:
            self.pipeline = Pipeline([('Scaler' , MinMaxScaler())])
    
        if is_train:
            data_paths_0 = sorted(glob.glob(dataset_path + '\\Normal\\*_0_*.csv'))
            data_paths_1 = sorted(glob.glob(dataset_path + '\\Normal\\*_1_*.csv'))
            data_paths_2 = sorted(glob.glob(dataset_path + '\\Normal\\*_2_*.csv'))
            data_paths_3 = sorted(glob.glob(dataset_path + '\\Normal\\*_3_*.csv'))

            self.data_paths = list(zip(data_paths_0, data_paths_1, data_paths_2, data_paths_3))
        else:
            data_paths_0 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_0_*.csv'))
            data_paths_1 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_1_*.csv'))
            data_paths_2 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_2_*.csv'))
            data_paths_3 = sorted(glob.glob(dataset_path + '\\Abnormal\\*_3_*.csv'))

            self.data_paths = list(zip(data_paths_0, data_paths_1, data_paths_2, data_paths_3))

    def __len__(self):
        return len(self.data_paths) // self.batch_size
        
    # def __call__(self) -> tf.data.Dataset:
    #     dataset = []
        
    #     for path_0, path_1, path_2, path_3 in self.data_paths:
    #         data_0 = pd.read_csv(path_0, encoding='unicode_escape', delimiter='\t', header=None)
    #         data_1 = pd.read_csv(path_1, encoding='unicode_escape', delimiter='\t', header=None)
    #         data_2 = pd.read_csv(path_2, encoding='unicode_escape', delimiter='\t', header=None)
    #         data_3 = pd.read_csv(path_3, encoding='unicode_escape', delimiter='\t', header=None)

    #         data_0 = np.array(data_0[0].values.tolist()[1:])
    #         data_1 = np.array(data_1[0].values.tolist()[1:])
    #         data_2 = np.array(data_2[0].values.tolist()[1:])
    #         data_3 = np.array(data_3[0].values.tolist()[1:])

    #         data_0 = np.mean(np.abs(data_0), axis=0).reshape(-1, 1)
    #         data_1 = np.mean(np.abs(data_1), axis=0).reshape(-1, 1)
    #         data_2 = np.mean(np.abs(data_2), axis=0).reshape(-1, 1)
    #         data_3 = np.mean(np.abs(data_3), axis=0).reshape(-1, 1)
            
    #         data = concate_data([data_0, data_1, data_2, data_3], axis=1)
    #         dataset.append(data)
        
    #     dataset = tf.data.Dataset.from_tensor_slices(dataset)

    #     dataset = dataset.batch(self.batch_size)

    #     if self.is_cache:
    #         dataset = dataset.cache()

    #     dataset = dataset.prefetch(self.AUTOTUNE)

    #     return dataset
    
    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]

        dataset_batch = self.data_paths[index]

        data_0 = pd.read_csv(dataset_batch[0], encoding='unicode_escape', delimiter='\t', header=None)
        data_1 = pd.read_csv(dataset_batch[1], encoding='unicode_escape', delimiter='\t', header=None)
        data_2 = pd.read_csv(dataset_batch[2], encoding='unicode_escape', delimiter='\t', header=None)
        data_3 = pd.read_csv(dataset_batch[3], encoding='unicode_escape', delimiter='\t', header=None)

        data_0 = np.array(data_0.values.tolist()[1:])
        data_1 = np.array(data_1.values.tolist()[1:])
        data_2 = np.array(data_2.values.tolist()[1:])
        data_3 = np.array(data_3.values.tolist()[1:])

        split_idx = int(len(data_0)*self.split[0])

        if self.train_mode == TrainMode.TRAIN.value:
            data_0 = data_0[:split_idx]
            data_1 = data_1[:split_idx]
            data_2 = data_2[:split_idx]
            data_3 = data_3[:split_idx]
        elif self.train_mode == TrainMode.VALIDATION.value:
            data_0 = data_0[split_idx:]
            data_1 = data_1[split_idx:]
            data_2 = data_2[split_idx:]
            data_3 = data_3[split_idx:]

        data_0 = np.mean(np.abs(data_0), axis=0).reshape(-1, 1)
        data_1 = np.mean(np.abs(data_1), axis=0).reshape(-1, 1)
        data_2 = np.mean(np.abs(data_2), axis=0).reshape(-1, 1)
        data_3 = np.mean(np.abs(data_3), axis=0).reshape(-1, 1)

        data_0 = data_0.reshape(-1, 1)
        data_1 = data_0.reshape(-1, 1)
        data_2 = data_0.reshape(-1, 1)
        data_3 = data_0.reshape(-1, 1)
        
        data = concate_data([data_0, data_1, data_2, data_3], axis=1)
        data = self.adjust_data(data)

        if self.is_lstm:
            data = data.reshape(1, 1, 4)

        self.cache[index] = data, data
        
        return data, data

    def adjust_data(self, data):        
        if self.is_adjust_fit:
            data = self.pipeline.fit_transform(data)
            self.is_adjust_fit = False
        else:
            data = self.pipeline.transform(data)

        return data