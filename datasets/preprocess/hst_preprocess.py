
from datasets.preprocess.utils import (
    setlogger, 
    normalize,
    generate_time_frequency_image_dataset, 
    loadmat_v73
)
import os
import logging
import numpy as np

from sklearn.utils import shuffle

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
        channel=13,
        time_steps=1024,
        overlap_ratio=0.5,
        normalization=False,
        random_seed=42,
        raw=False,
        fft=True
):
    logging.info("Domain: {}, normalization: {}, time_steps: {}, overlap_ratio: {}."
                .format(domain, normalization, time_steps, overlap_ratio))
    
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


def extract_dict_data(dataset):
    x = np.concatenate([dataset[key] for key in dataset.keys()])
    y = []
    for i, key in enumerate(dataset.keys()):
        number = len(dataset[key])
        y.append(np.tile(i, number))
    y = np.concatenate(y)
    return x, y


if __name__ == '__main__':
    # Data Splitting Parameters
    dir_path = './data'
    dataset_name = 'HST'
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
    for i in range(3):
        dataset = load_HST_dataset(i, './data', 13, 2048)
        img_dir = dir_path + "/STFTImageData_HST/" + str(i) + "/"
        generate_time_frequency_image_dataset(
            dataset_name,
            dataset, 
            labels,
            image_size, 
            window_size, 
            overlap, 
            img_dir)