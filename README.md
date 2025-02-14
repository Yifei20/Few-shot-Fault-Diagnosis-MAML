# MAML Application on Few-shot Fault Diagnosis (PyTorch)

## 1. Introduction

This project employs the **Model-Agnostic Meta-Learning** (MAML) framework to address the **Cross-domain Few-shot Fault Diagnosis problem**, specifically on the *[Case Western Reserve University (CWRU) bearing fault dataset](https://engineering.case.edu/bearingdatacenter)* and a *closed-source high-speed train (HST) fault dataset*. Since HST is not open source, this repository focuses on applications and results in the CWRU database. It implements this framework under different cross-domain settings, including  source domains consisting of tasks from one or more working conditions and different few-shot learning settings. This project also implements three different methods for preprocessing the raw signal data into 1-D or 2-D data for classification. The pre-processed 1-D or 2-D data are classified using different CNN-based models.

### 1.1 Cross-domain Few-shot Learning

**Few-shot** describes a class of problems, for which we usually use $N$-way $K$-shot to describe the problem setting. Where $N$ denotes the total number of classes we want the model to discriminate between, and $K$ denotes the number of labelled samples for each class that the model can acquire for training. That is namely, our model receives a total of $N \times K$ data as input for training, after which it is tested with new samples from these $N$ classes. We want the model to be able to achieve high performance using as few as possible $K$ number of samples, usually 1 or 5.

**Cross-domain** typically refers to a scenario where the training data (source domain) and the testing data (target domain) have different distributions. In the scenario of bearing fault diagnosis, bearing data under different loads (working conditions) can be treated as different domains. Therefore, **cross-domain few-shot learning** describes a kind of task that we have enough labelled training data in one or more domains, while in some domains, we only have very limited labelled data. We want to transfer the knowledge learned from domains with many labelled training data to the domain with limited labelled data.

### 1.2 CWRU Dataset and Preprocessing

In this project, the fault categories of the CWRU dataset are categorized into a total of 10 types of data, including one type of health data and nine types of fault data, as shown in the table below. The sampling frequency of these data is 12k Hz, except for the normal data, which is sampled at 48k Hz. 

Then, a sliding window of length 1024 with an overlap rate of 0.5 was used to sample the original data and construct the raw data samples. Subsequently, three preprocessing methods, including **Fast Fourier Transform** (FFT) that generates one-dimensional data and **Short-Time Fourier Transform** (STFT) and **Wavelet Transform** (WT) that generate two-dimensional Time-Frequency Images (TFIs) were used to preprocess these raw samples.

![CWRU fault label encoding method](https://s2.loli.net/2025/02/15/7KHQZbmIz52RWj9.png)

### 1.3 First-order Approximation

Due to the computational resource requirements of MAML during training, this project also includes the option to use first-order approximation to improve the computational performance of the model. After testing, enabling first-order approximation does not seem to significantly degrade the model performance, you can easily choose whether to enable this feature or not by passing a parameter when calling function.

## 2. Experiments

This section shows the experimental results of MAML's cross-domain few-shot learning on the CWRU dataset. Two cross-domain settings settings were tested and compared. Two few-shot settings were tested for each cross-domain setting.

### 2.1 Comparison of Single and Multiple Source Domains

As the results of a literature review, previous experiments using MAML for cross-domain fault diagnosis have basically only considered the case of single source domain, which is not consistent with the core concept of MAML. MAML is designed to generalise from many *different tasks* (with different distributions) so that it makes our model able to learn quickly for a new task. For example, we may probably train MAML using some binary classification tasks, like cat-dog, apple-pear and horse-donkey classification tasks. Then our model may be able to adapt quickly to monkey-gorilla binary classification task using very limited training data (maybe only one for each). While single source domain cannot give our MAML framework enough task distributions to generalise, which prevents MAML from performing at its true capacity. Therefore, we conducted two cross-domain experiment settings aiming to prove the limitations of the previous works.

### 2.2 Experiment Results

The results of the mentioned single and multiple source domain settings are listed in the table below, where *Single* denotes the former and *Multiple* denotes the latter. In CWRU dataset, we have four working conditions in total depending on the load (from 0 to 3), denoted as $D_i$, where $i$ denotes for the index of the domain. All of the three loaded working conditions ($D_1,D_2,D_3$) were used as target domains respectively, the no-loaded working condition ($D_0$) was only used as source domain. 

If the experiment is single source domain and $D_1$ is the target domain, then the other three domains ($D_0, D_2, D_3$) were used as source domain respectively and averaged as the final result. If the experiment is multiple source domains $D_1$ is the target domain, then all other three domains were used together as source domain. Target domain $D_i$ was used to express the specific comparison setting. This experiment was conducted on both 10-way 1-shot and 10-way 5-shot settings, only shot numbers are indicated in the table due to the length limitation of Markdown.

| # of Source Domain | $D_1$ 1-shot | $D_1$ 5-shot | $D_2$ 1-shot | $D_2$ 5-shot | $D_3$ 1-shot | $D_3$ 5-shot |
| ----------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Single     | 93.97        | 95.36        | 95.41        | 98.31        | 95.31        | 96.25        |
| Multiple   | 98.43        | 99.56        | 99.68        | 99.91        | 98.75        | 99.74        |

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

3. Train and test MAML on the obtained dataset, you can run `python train_maml -h` to see the detailed parameter setting options.

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
