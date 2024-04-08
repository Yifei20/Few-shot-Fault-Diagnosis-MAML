import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
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



def generate_time_frequency_image_dataset(dataset, labels, img_size, window_size, overlap, img_dir):
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