import numpy as np
import tensorflow as tf
import json

import enum
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from random import randint

class TrainMode(enum.Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

class DataGenerator():
    def __init__(self, 
    dataset_path,
    seed,
    split:tuple, 
    num_limit : int,
    caching=False,
    shuffle=True, 
    batch_size=64, 
    predict_only :bool = False,
    ):
        self.dataset_path = dataset_path
        self.file_paths = []
        self.labels = []
        self.seed = seed
        self.shuffle = shuffle
        self.split = split
        self.batch_size = batch_size
        self.caching = caching
        self.predict_only = predict_only
        self.num_limit = num_limit

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.train_paths = None
        self.train_labels = None
        self.validation_paths = None
        self.validation_labels = None
        self.test_paths = None
        self.test_labels = None

        if not self.predict_only:
            with open(self.dataset_path, 'r') as file:
                data = json.load(file)
                data2 = data['abnormal_feature'] 
                data = data['normal_feature']

            if self.num_limit:
                np.random.shuffle(data)
                data = data[:self.num_limit]

                for i in range(self.num_limit):
                    self.labels.append(1)

            self.file_paths.extend(data)

            train_size = 1-sum(self.split[1:])
            
            if self.split[0] == 1:
                self.train_paths = self.file_paths
                self.train_labels = self.labels
            elif self.split[1] == 1:
                self.validation_paths = self.file_paths
                self.validation_labels = self.validation_labels
                validation_size = 0.0
            elif self.split[2] == 1:
                self.test_paths = self.file_paths[3600:]
                self.test_labels = self.labels[3600:]

            if self.split[0] > 0 and self.split[0] != 1:
                self.train_paths, self.validation_paths, self.train_labels, self.validation_labels = train_test_split(
                                                                                            self.file_paths,
                                                                                            self.labels,
                                                                                            train_size=train_size,
                                                                                            stratify=self.labels,
                                                                                            random_state=self.seed)

            if self.split[2] > 0 and self.split[2] != 1:
                validation_size = (self.split[1] / self.split[2])/((self.split[1] / self.split[2]) + (self.split[2] / self.split[1]))
                self.validation_paths, self.test_paths, self.validation_labels, self.test_labels = train_test_split(
                                                                                                    self.validation_paths,
                                                                                                    self.validation_labels,
                                                                                                    train_size=validation_size,
                                                                                                    stratify=self.validation_labels,
                                                                                                    random_state=self.seed)

            self.test_paths += data2
            for i in range(len(data2)):
                self.test_labels.append(0)

        else:
            self.file_paths = tf.io.gfile.glob(f'{self.dataset_path}/*')

    def to_dataset(self, paths, labels):
        # 데이터 백터화를 통해 연산 속도 햘상
        series_ds = tf.data.Dataset.from_tensor_slices(paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)

        return tf.data.Dataset.zip((series_ds, label_ds))

    def __call__(self, train_mode) -> tf.data.Dataset:
        self.train_mode = train_mode

        if self.train_mode == 'train':
            self.dataset = self.to_dataset(self.train_paths, self.train_labels)
        elif self.train_mode == 'val':
            self.dataset = self.to_dataset(self.validation_paths, self.validation_labels)
        elif self.train_mode == 'test':
            self.dataset = self.to_dataset(self.test_paths, self.test_labels)
        elif self.train_mode == 'predict':
            self.dataset = self.to_dataset(self.file_paths)

        if self.batch_size:
            self.dataset = self.dataset.batch(self.batch_size) 

        if self.caching:
            self.dataset = self.dataset.cache()
        
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return self.dataset