# MAML Application on Few-shot Fault Diagnosis (PyTorch)

## 1. Introduction

This project employs the Model-Agnostic Meta-Learning (MAML) framework to address the cross-domain few-shot fault diagnosis problem, specifically on the *[Case Western Reserve University (CWRU) bearing fault dataset](https://engineering.case.edu/bearingdatacenter)* and a *closed-source high-speed train (HST) fault dataset*. Since HST is not open source, this repository focuses on applications and results in the CWRU database. It implements this framework under different cross-domain settings, including  source domains consisting of tasks from one or more working conditions and different few-shot learning settings. This project also implements different methods for preprocessing the raw signal data, including *Fast Fourier Transform (FFT)* that generates one-dimensional data and *Short-Time Fourier Transform (STFT)* and *Wavelet Transform (WT)* that generate two-dimensional *Time-Frequency Images (TFIs)*. The processed 1D and 2D data are classified using different CNN-based models.

### 1.1 CWRU Dataset and Preprocessing

In this project, the fault categories of the CWRU dataset are categorized into a total of 10 types of data, including one type of health data and nine types of fault data, as shown in the following table. The sampling frequency of these data is 12k Hz, except for the normal data, which is sampled at 48k Hz. Then, a sliding window of length 1024 with an overlap rate of 0.5 was used to sample the original data and construct the raw data samples. Subsequently, three preprocessing methods (FFT, STFT and WT) were used to preprocess these raw samples.

![CWRU fault label encoding method](https://s2.loli.net/2024/10/29/yJLcsFkm2wBVbdZ.png)

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

This implementation is based on Python 3, and the detailed package requirements of this project are listed below. You can install this requirement using `requirements.txt` file directly by running the following command in your terminal.

```shell
pip install -r requirements.txt
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
    - Download the raw data:
      ```shell
      mkdir data
      cd data
      wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/raw_dataset/CWRU_12k.zip
      unzip CWRU_12k.zip
      cd ..
      ```
   - If you want to use 2D TFIs as the input to the framework, you can download the preprocessed (by [STFT](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/tag/stft_dataset) and [WT](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/tag/wt_dataset)) data using the following command
     ```shell
     cd data
     wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/wt_dataset/WT_CWRU.zip
     wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/stft_dataset/STFT_CWRU.zip
     unzip WT_CWRU.zip
     unzip STFT_CWRU.zip
     cd ..
     ```
     or process the data yourself using the `preprocess_cwru.py` (very time consuming). If you want to change the preprocessing settings, you can modify the parameter settings in its main function.
     ```shell
     python preprocess_cwru.py
     ```

4. Train and test MAML on the obtained dataset, you can run `python train_maml -h` to see the detailed parameter setting options.
    - For example, if you want to train model on domain 0,1,2 and then transfer it to domain 3, you can run it using the following command:
      ```shell
      python train_maml.py --ways 10 --shots 5 --iter 1000 --first_order True --preprocess FFT --train_domains 0,1,2 --test_domain 3
      ```
    - Or, if you want to train model on domain 0 and then transfer it to domain 3, you can just change 0,1,2 to 0, like
      ```shell
      python train_maml.py --ways 10 --shots 5 --iter 1000 --first_order True --preprocess FFT --train_domains 0 --test_domain 3
      ```

## 5. References

- [Xiaohan-Chen/few-shot-fault-diagnosis](https://github.com/Xiaohan-Chen/few-shot-fault-diagnosis)
- [learnables/learn2learn](https://github.com/learnables/learn2learn)
- [Yifei20/MAML-PyTorch](https://github.com/Yifei20/MAML-PyTorch)
