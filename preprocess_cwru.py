import numpy as np
import logging
import os

from scipy.io import loadmat
from sklearn.utils import shuffle
from utils import (
    setlogger, 
    normalize,
    generate_time_frequency_image_dataset
)


# Name dictionary of different fault types in each working condition
dataname_dict= {
    0:[97, 105, 118, 130, 169, 185, 197, 209, 222, 234],  # load 0 HP, motor speed 1797 RPM
    1:[98, 106, 119, 131, 170, 186, 198, 210, 223, 235],  # load 1 HP, motor speed 1772 RPM
    2:[99, 107, 120, 132, 171, 187, 199, 211, 224, 236],  # load 2 HP, motor speed 1750 RPM
    3:[100, 108, 121, 133, 172, 188, 200, 212, 225, 237]  # load 3 HP, motor speed 1730 RPM
    }  
# Partial part of the axis name
axis = "_DE_time"
# Labels of different fault types
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_CWRU_dataset(
        domain, 
        dir_path, 
        time_steps=1024,
        overlap_ratio=0.5,
        normalization=False,
        random_seed=42,
        raw=False,
        fft=True
):
    logging.info("Domain: {}, normalization: {}, time_steps: {}, overlap_ratio: {}."
                 .format(domain, normalization, time_steps, overlap_ratio))

    # dataset {class label : data list of this class}
    # e.g., {0: [data1, data2, ...], 1: [data1, data2, ...], ...}
    dataset = {label: [] for label in labels}

    for label in labels:
        fault_type = dataname_dict[domain][label]
        if fault_type < 100:
            realaxis = "X0" + str(fault_type) + axis
        else:
            realaxis = "X" + str(fault_type) + axis
        data_path = dir_path + "/CWRU_12k/Drive_end_" + str(domain) + "/" + str(fault_type) + ".mat"
        mat_data = loadmat(data_path)[realaxis].reshape(-1)
        if normalization:
            mat_data = normalize(mat_data)
        
        # Total number of samples is calculated automatically. No need to set it manually.
        stride = int(time_steps * (1 - overlap_ratio))
        sample_number = (len(mat_data) - time_steps) // stride + 1
        logging.info("Loading Data: fault type: {}, total num: {}, sample num: {}"
                     .format(label, mat_data.shape[0], sample_number))
        # sample_number = 20 # for testing

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
    dataset_name = 'CWRU'
    algorithm = 'WT' # 'STFT' or 'WT'
    time_steps = 1024
    overlap_ratio = 0.5
    # STFT Parameters
    image_size = 84
    window_size = 64
    overlap = 0.5

    # Set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/preprocess.log")

    for i in range(4):
        dataset = load_CWRU_dataset(i, './data')
        img_dir = dir_path + "/{}_{}/Drive_end_".format(algorithm, dataset_name) + str(i) + "/"
        generate_time_frequency_image_dataset(
            dataset_name,
            algorithm,
            dataset, 
            labels,
            image_size, 
            window_size, 
            overlap, 
            img_dir)