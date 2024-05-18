
from utils import (
    setlogger, 
    normalize,
    generate_time_frequency_image_dataset, 
    loadmat_v73
)
import os
import logging
import numpy as np

from sklearn.utils import shuffle

dataname_dict= {
    0:['D00AA', 'Dα7BA', 'Dα7JA', 'Dα7UA', 'Dβ7BA', 'Dβ7JA', 'Dβ7UA', 'Dγ7BA', 'Dγ7JA', 'Dγ7UA'],  # A: 20km/h
    1:['D00AH', 'Dα7BH', 'Dα7JH', 'Dα7UH', 'Dβ7BH', 'Dβ7JH', 'Dβ7UH', 'Dγ7BH', 'Dγ7JH', 'Dγ7UH'],  # H: 160km/h
    2:['D00AN', 'Dα7BN', 'Dα7JN', 'Dα7UN', 'Dβ7BN', 'Dβ7JN', 'Dβ7UN', 'Dγ7BN', 'Dγ7JN', 'Dγ7UN'],  # N: 280km/h
    }
axis_front = "D_7"
axis_end = "_30S_"

def load_HST_dataset(
        domain,
        dir_path,
        partial=True,
        labels=[0, 2, 3, 5, 6],
        channel=13,
        time_steps=1024,
        overlap_ratio=0.5,
        normalization=False,
        random_seed=42,
        raw=False,
        fft=True
):
    if not partial:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    logging.info("Loading HST dataset...\n \
                 Loading data domain: {}, if raw: {}, if partial: {},\n \
                 Labels: {}, chnnel {}, time_steps: {}, overlap_ratio: {},\n \
                 If using normalization: {}, if using FFT (if raw) {}."
                .format(domain, raw, partial, labels, channel+1, time_steps, overlap_ratio, normalization, fft))
    
    dataset = {label: [] for label in labels}

    for label in labels:
        data_path = dir_path + "/HST/" + str(domain) + "/" + dataname_dict[domain][label] + ".mat"
        if label == 0:
            realaxis = dataname_dict[domain][label]
        elif label > 0 and label < 4:
            realaxis = dataname_dict[domain][label][3:] + axis_end
        else:
            if (domain == 0 or domain == 1) and label == 6:
                realaxis = axis_front + dataname_dict[domain][label][3:] + "_33S_"
            else:
                realaxis = axis_front + dataname_dict[domain][label][3:] + axis_end
        
        mat_data = loadmat_v73(data_path, realaxis, channel)
        if label != 0:
            mat_data = mat_data[int(len(mat_data) * 0.55):]
        if normalization:
            mat_data = normalize(mat_data)

        # Total number of samples is calculated automatically. No need to set it manually.
        stride = int(time_steps * (1 - overlap_ratio))
        sample_number = (len(mat_data) - time_steps) // stride + 1
        logging.info("Loading Data: fault type: {}, total num: {}, sample num: {}"
                     .format(label, mat_data.shape[0], sample_number))
        
        for i in range(sample_number):
            start = i * stride
            end = start + time_steps
            sub_data = mat_data[start : end]
            if raw:
                sub_data = sample_preprocessing(sub_data, fft)
            dataset[label].append(sub_data)
        # Shuffle the data
        dataset[label] = shuffle(dataset[label], random_state=random_seed)
        logging.info("Data is shuffled using random seed: {}\n"
                     .format(random_seed))
    return dataset


def sample_preprocessing(sub_data, fft):
    if fft:
        sub_data = np.fft.fft(sub_data)
        sub_data = np.abs(sub_data) / len(sub_data)
        sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1,)           
    sub_data = sub_data[np.newaxis, :]

    return sub_data


if __name__ == '__main__':
    # Data Splitting Parameters
    dir_path = './data'
    labels = [0, 2, 3, 5, 6]
    dataset_name = 'HST'
    channel=13
    algorithm = 'WT' # 'STFT' or 'WT'
    time_steps = 2048
    overlap_ratio = 0.5
    # STFT Parameters
    image_size = 84
    window_size = 64
    overlap = 0.5

    # Set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/preprocess.log")
    for i in range(3):
        dataset = load_HST_dataset(i, './data', partial=True, labels=labels, channel=13, time_steps=time_steps)
        img_dir = dir_path + "/{}_{}/".format(algorithm, dataset_name) + str(i) + "/"
        generate_time_frequency_image_dataset(
            dataset_name,
            algorithm,
            dataset, 
            labels,
            image_size, 
            window_size, 
            overlap, 
            img_dir)