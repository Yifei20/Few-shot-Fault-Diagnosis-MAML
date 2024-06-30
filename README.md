# MAML Application on Few-shot Fault Diagnosis (PyTorch)

## 1. Introduction

This project employs the Model-Agnostic Meta-Learning (MAML) framework to address the cross-domain few-shot fault diagnosis problem, specifically on the *[Case Western Reserve University (CWRU) bearing fault dataset](https://engineering.case.edu/bearingdatacenter)* and a *[closed-source high-speed train (HST) fault dataset](http://www.aas.net.cn/article/zdhxb/2019/12/2218)*. Since HST dataset is not open source, this repository mainly introduces the application and results on CWRU dataset.

It implements this framework under different cross-domain settings, including  source domains consisting of tasks from one or more working conditions and different few-shot learning settings. This project also implements different methods for preprocessing the raw signal data, including *Fast Fourier Transform (FFT)* that generates one-dimensional data and *Short-Time Fourier Transform (STFT)* and *Wavelet Transform (WT)* that generate two-dimensional *Time-Frequency Images (TFIs)*. The processed 1D and 2D data are classified using different CNN-based models.

### 1.1 CWRU Dataset and Preprocessing

In this project, the fault categories of the CWRU dataset are categorized into a total of 10 types of data, including one type of health data and nine types of fault data, as shown in the following table. The sampling frequency of these data is 12k Hz, except for the normal data, which is sampled at 48k Hz.

In this project, a sliding window of length 1024 with an overlap rate of 0.5 was used to sample the original data and construct the raw data samples. The three subsequent preprocessing methods (FFT, STFT and WT) are based on the samples obtained from this sampling method.

![CWRU-fault-type-table](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/202405311753463.png)

### 1.2 Cross-domain Setting

This experiment sets up a variety of cross-domain few-shot learning settings. In particular, it compares two cross-domain learning settings: One is to use the set of meta-tasks sampled from multiple working conditions as the source domain and another specific working condition as the target domain; the other is to use the source meta-tasks sampled from only one specific working condition as the source domain, with another different working condition as the target domain.

The second cross-domain learning setting is easy to understand and is not described additionally here. For the first one, it is actually the case that first we will choose a few working conditions as the source domain and in addition a different working condition as the target domain. Then, during each meta-training process, we randomly select one working condition from the source domains to sample the meta-tasks for meta-training. The rest of the settings are the same as the second one.

### 1.3 First-order Approximation

Due to the computational resource requirements of MAML during training, this project also includes the option to use first-order approximation to improve the computational performance of the model. After testing, enabling first-order approximation does not seem to significantly degrade the model performance, and the user can easily choose whether to enable this feature or not.

## 2. Experiment Results

This section shows the experimental results of MAML's cross-domain few-shot learning on the CWRU dataset, and the following table compares the case where the source domain contains only one working  domain (MAML) and multiple working domains (MDML).

In this case, each of the three loaded working conditions is tested experimentally as a target domain, which is denoted as $D_i$, where $i$ stands for the number of the working condition.
For MAML case, this project uses the no-loaded working conditions and other loaded working conditions except the target working conditions in turn as the source domain for training and testing, and the final results are averaged.
For MDML case, the project directly uses all the conditions except the target condition as the source domain for training and testing, and obtains the results. This experiment was conducted on both 10-way 1-shot and 10-way 5-shot settings.

| Methods | $D_1$ 1-shot | $D_1$ 5-shot | $D_2$ 1-shot | $D_2$ 5-shot | $D_3$ 1-shot | $D_3$ 5-shot |
| ------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| MAML    | 93.97        | 95.36        | 95.41        | 98.31        | 95.31        | 96.25        |
| MDML    | 98.43        | 99.56        | 99.68        | 99.91        | 98.75        | 99.74        |

## 3. Requirements

This implementation is based on Python 3, and the detailed requirements of this project are listed below. You can install this requirement using `requirement.txt` file directly by the following code in terminal.

```shell
pip install requirement.txt
```

- h5py==3.2.1
- learn2learn==0.2.0
- matplotlib==3.4.3
- numpy==1.22.0
- Pillow==10.3.0
- PyWavelets==1.1.1
- scikit_learn==0.24.2
- scipy==1.7.1
- torch==2.1.1
- torchvision==0.16.1

## 4. Usage

To run this code, you can follow the instructions below.

1. Clone this repo.

```shell
git clone https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML
cd Few-shot-Fault-Diagnosis-MAML
```

2. Download the [CWRU Bearing Fault](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/tag/raw_dataset) dataset.

```shell
mkdir data
cd data
wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/raw_dataset/CWRU_12k.zip
unzip CWRU_12k.zip
```

If you want to use 2D TFIs as the input to the framework, you need to preprocess the raw data using STFT or WT. Or, you can directly download the preprocessed TFI data from [STFT](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/tag/stft_dataset), [WT](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/tag/wt_dataset). The downloading process is the same as step 2.

*If you want to change the preprocessing settings, you can modify the parameter settings first in this file's main function.*

3. (Optional) Preprocess the raw data to transform them into TFIs.

```shell
python preprocess_cwru.py
```

4. Train MAML on the obtained dataset (please refer to the *train_maml.py* to see more detailed parameter settings).

```shell
python train_maml.py --ways 10 --shots 5 --iter 1000 --first_order True --preprocess FFT --train_domains 0,1,2 --test_domain 3
```

## 5. References

- [Xiaohan-Chen/few-shot-fault-diagnosis](https://github.com/Xiaohan-Chen/few-shot-fault-diagnosis)
- [learnables/learn2learn](https://github.com/learnables/learn2learn)
- [Yifei20/MAML-PyTorch](https://github.com/Yifei20/MAML-PyTorch)
