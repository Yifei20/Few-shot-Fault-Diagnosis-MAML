import numpy as np
import matplotlib.pyplot as plt
import logging
import sklearn
import os

from scipy.io import loadmat
from scipy.signal import stft
from PIL import Image
from utils import setlogger


# Name dictionary of different fault types in each working condition
dataname_dict= {0:[97, 105, 118, 130, 169, 185, 197, 209, 222, 234],  # load 0 HP, motor speed 1797 RPM
                1:[98, 106, 119, 131, 170, 186, 198, 210, 223, 235],  # load 1 HP, motor speed 1772 RPM
                2:[99, 107, 120, 132, 171, 187, 199, 211, 224, 236],  # load 2 HP, motor speed 1750 RPM
                3:[100, 108, 121, 133, 172, 188, 200, 212, 225, 237]}  # load 3 HP, motor speed 1730 RPM
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
        random_seed=42
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
            dataset[label].append(sub_data)

        # Shuffle the data
        dataset[label] = sklearn.utils.shuffle(dataset[label], random_state=random_seed)
        logging.info("Data is shuffled using random seed: {}\n".format(random_seed))
    
    return dataset

def normalize(data):
    return (data-min(data)) / (max(data)-min(data))


def make_time_frequency_image(dataset, img_size, window_size, overlap, img_path):
    overlap_samples = int(window_size * overlap)
    frequency, time, magnitude = stft(dataset, nperseg=window_size, noverlap=overlap_samples)
    magnitude = np.abs(magnitude)

    # Image Plotting
    plt.pcolormesh(time, frequency, magnitude, shading='gouraud')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gcf().set_size_inches(img_size/100, img_size/100)
    plt.savefig(img_path, dpi=100)
    plt.clf()
    plt.close()

def generate_time_frequency_image_dataset(dataset, img_size, window_size, overlap, img_dir):
    for label in labels:
        count = 0
        for i, data in enumerate(dataset[label]):
            os.makedirs(img_dir, exist_ok=True)
            img_path = img_dir + str(label) + "_" + str(count)
            make_time_frequency_image(data, img_size, window_size, overlap, img_path)
            count += 1
    image_list = os.listdir(img_dir)
    for image_name in image_list:
        image_path = os.path.join(img_dir, image_name)
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.save(image_path)


if __name__ == '__main__':
    # Data Splitting Parameters
    dir_path = './data'
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
        img_dir = dir_path + "/STFTImageData/Drive_end_" + str(i) + "/"
        generate_time_frequency_image_dataset(
            dataset, 
            image_size, 
            window_size, 
            overlap, 
            img_dir)