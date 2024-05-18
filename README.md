# MAML Application on Few-shot Fault Diagnosis (PyTorch)

This repository employs the MAML framework to address the cross-domain few-shot fault diagnosis problem, specifically on the *[CWRU bearing fault dataset](https://engineering.case.edu/bearingdatacenter)* and a *closed-source high-speed train (HST) fault dataset*.

## Implementation Description

### Dataset Introduction

The fault categories of the CWRU dataset are summarized into a total of ten categories of data, including one type of normal data and nine types of fault data, which are presented in the table below. The dataset contains two kinds of data with sampling frequencies of 12k Hz and 48k Hz, with the exception of the normal data, which only has a sampling frequency of 48k Hz. For this data, we utilize the data with a sampling frequency of 12k Hz.

![CWRU-fault-type-table](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/cwru-table-202405181930931.png)

### Data Pre-processing

It implements 3 data-preprocessing methods, including Fast Fourier Transform (FFT), Short-time Fourier Transform (STFT) and Wavelet Transform (WT). STFT and WT are used to generate 2D time-frequency images (TFIs) to further express data features for a 2D CNN base model. FFT is used to process the 1D data for a 1D CNN base model. 

### Cross-domain Setting

It offers a option to distribute the source domain and target domain for the cross-domain setting. In particular, this implementation provides with an option of using multiple working conditions as its source domains or just choosing one as the source domain.

### First-order Approximation

It also has an ability to enable the first-order approximation function for the original MAML framework to enhance the computational performance of the model. User can easily choose whether to enable it or not.

### Basic Framework

This implementation uses the third-party Python library [learn2learn](https://github.com/learnables/learn2learn/), which has already wrapped MAML in it. If you are interested in the base algorithm of MAML, you can see my previous MAML implementation: [MAML-PyTorch](https://github.com/Yifei20/MAML-PyTorch).

## Experiment Results

This part contains the experiment results of MAML on CWRU bearing fault dataset using one or multiple source domains. The table below shows the results of this experiment, in which MAML indicates the original MAML using one operational condition as the source domain, while MDML denotes using multiple source domains at the same time. In addition, $D_i$ denotes the average performance of MAML for using different working conditions as the source domain and the performance of using many conditions as the source domain simultaneously. This experiment was conducted on both 10-way 1-shot and 10-way 5-shot settings.

| Methods | $D_1$ 1-shot | $D_1$ 5-shot | $D_2$ 1-shot | $D_2$ 5-shot | $D_3$ 1-shot | $D_3$ 5-shot |
| ------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| MAML    | 93.97        | 95.36        | 95.41        | 98.31        | 95.31        | 96.25        |
| MDML    | 98.43        | 99.56        | 99.68        | 99.91        | 98.75        | 99.74        |

## Usage

To run this code, you can follow the instructions below.

1. Clone this repo.

```shell
git clone https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML
cd Few-shot-Fault-Diagnosis-MAML
```

2. Download the [CWRU Bearing Fault]() dataset.

```shell
mkdir data
cd data
wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/dataset/CWRU_12k.zip
unzip CWRU_12k.zip
```

3. Pre-process the original data to transform them into TFIs (optional).

```shell
python preprocess_cwru.py # You can modify the parameter settings first in this file's main function
```

4. Train MAML on the obtained dataset (please refer to the train_maml.py to see more detailed parameter settings).

```shell
python train_maml.py --ways 10 --shots 5 --iter 1000 --first_order True
```

## References

- [Xiaohan-Chen/few-shot-fault-diagnosis](https://github.com/Xiaohan-Chen/few-shot-fault-diagnosis)

- [Yifei20/MAML-PyTorch](https://github.com/Yifei20/MAML-PyTorch)
