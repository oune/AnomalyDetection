import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tensorflow as tf

def get_mel_spectrogram_with_librosa(data, frame_length : int = 0.064, frame_stride : int = 0.032, num_mels : int = 39, sample_rate : int = 4000):    
    data = data.astype('float32')
    if len(data.shape) != 1:
        data = data.squeeze(axis=1)
    
    input_nfft = int(round(sample_rate*frame_length)) # window length
    input_stride = int(round(sample_rate*frame_stride)) # window step

    mel_spectrogram = librosa.feature.melspectrogram(y=data, n_mels=num_mels, n_fft=input_nfft, hop_length=input_stride)

    return mel_spectrogram

def get_mel_spectrogram_with_tf(data, frame_length: int = 1024, frame_step : int = 512, num_mel_bins : int = 39, sample_rate : int = 160000):
    stfts = tf.signal.stft(
        data, frame_length=frame_length, frame_step=frame_step
        )

    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                                            num_mel_bins, 
                                                                            num_spectrogram_bins, 
                                                                            sample_rate
                                                                        )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    return mel_spectrograms

def stack_feature(sensor_0, sensor_1, sensor_2, sensor_3):
    feature_list = []

    for data_0, data_1, data_2, data_3 in zip(sensor_0, sensor_1, sensor_2, sensor_3):
        data_0 = get_mel_spectrogram_with_librosa(data_0)
        data_1 = get_mel_spectrogram_with_librosa(data_1)
        data_2 = get_mel_spectrogram_with_librosa(data_2)
        data_3 = get_mel_spectrogram_with_librosa(data_3)
        
        feature = np.stack([data_0, data_1, data_2, data_3])
        feature = feature.swapaxes(0, 1)
        feature = feature.swapaxes(1, 2)
        feature_list.append(feature)
    
    return np.array(feature_list)

def display_feature(feature, frame_stride : int = 0.032, sample_rate : int = 4000):
    input_stride = int(round(sample_rate*frame_stride))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(feature, ref=np.max), y_axis='mel', sr=sample_rate, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()