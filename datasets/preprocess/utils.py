import logging
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pywt
from PIL import Image



def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=path)
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)



def normalize(data):
    return (data-min(data)) / (max(data)-min(data))



def make_time_frequency_image_STFT(dataset_name, 
                              dataset, 
                              img_size, 
                              window_size, 
                              overlap, 
                              img_path):

    overlap_samples = int(window_size * overlap)
    
    frequency, time, magnitude = stft(dataset, nperseg=window_size, noverlap=overlap_samples)
    
    if dataset_name == 'HST':
        magnitude = np.log10(np.abs(magnitude) + 1e-10)
    else:
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


def make_time_frequency_image_WT(dataset_name,
                                 data,
                                 img_size,
                                 img_path):
    # Data Length
    sampling_length = len(data)
    # Wavelet Transform Parameters Setting
    if dataset_name == 'CWRU':
        sampling_period  = 1.0 / 12000
        total_scale = 128
        wavelet = 'cmor100-1'
    elif dataset_name == 'HST':
        sampling_period = 4e-6
        total_scale = 16
        wavelet = 'morl'
    else:
        raise ValueError("Invalid dataset name")
    fc = pywt.central_frequency(wavelet)
    cparam = 2 * fc * total_scale
    scales = cparam / np.arange(total_scale, 0, -1)
    # Conduct Wavelet Transform
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period)
    amplitude = abs(coefficients)
    if dataset_name == 'HST':
        amplitude = np.log10(amplitude + 1e-4)
    # Image Plotting
    t = np.linspace(0, sampling_period, sampling_length, endpoint=False)
    plt.contourf(t, frequencies, amplitude, cmap='jet')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gcf().set_size_inches(img_size/100, img_size/100)
    plt.savefig(img_path, dpi=100)
    plt.clf()
    plt.close()


def generate_time_frequency_image_dataset(dataset_name, 
                                          algorithm,
                                          dataset, 
                                          labels, 
                                          img_size, 
                                          window_size, 
                                          overlap, 
                                          img_dir):
    for index in range(len(labels)):
        count = 0
        for i, data in enumerate(dataset[labels[index]]):
            os.makedirs(img_dir, exist_ok=True)
            img_path = img_dir + str(index) + "_" + str(count)
            if algorithm == 'STFT':
                make_time_frequency_image_STFT(dataset_name, 
                                               data, 
                                               img_size, 
                                               window_size, 
                                               overlap, 
                                               img_path)
            elif algorithm == 'WT': 
                make_time_frequency_image_WT(dataset_name, 
                                             data, 
                                             img_size, 
                                             img_path)
            else:
                raise ValueError("Invalid algorithm name")
            count += 1
    image_list = os.listdir(img_dir)
    for image_name in image_list:
        image_path = os.path.join(img_dir, image_name)
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.save(image_path)


def loadmat_v73(data_path, realaxis, channel):
    with h5py.File(data_path, 'r') as f:
        mat_data = f[f[realaxis]['Y']['Data'][channel][0]]
        return mat_data[:].reshape(-1)