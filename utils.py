import logging
import os
import torch
import pywt
import logging
import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import stft
from PIL import Image



def setup_logger(log_path, experiment_title):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(os.path.join(log_path, 
                                          experiment_title + '.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def print_logs(iteration, meta_train_error, meta_train_accuracy, meta_test_error, meta_test_accuracy):
    logging.info('Iteration {}:'.format(iteration))
    logging.info('Meta Train Results:')
    logging.info('Meta Train Error: {}.'.format(meta_train_error))
    logging.info('Meta Train Accuracy: {}.'.format(meta_train_accuracy))
    logging.info('Meta Test Results:')
    logging.info('Meta Test Error: {}.'.format(meta_test_error))
    logging.info('Meta Test Accuracy: {}.\n'.format(meta_test_accuracy))

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
    

def extract_dict_data(dataset):
    x = np.concatenate([dataset[key] for key in dataset.keys()])
    y = []
    for i, key in enumerate(dataset.keys()):
        number = len(dataset[key])
        y.append(np.tile(i, number))
    y = np.concatenate(y)
    return x, y



if __name__ == '__main__':
    pass