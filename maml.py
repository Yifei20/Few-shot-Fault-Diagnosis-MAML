from datasets.cwru import CWRU

import logging
import random
import numpy as np
import torch
from torch import nn
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

import matplotlib.pyplot as plt
from utils import (setlogger, fast_adapt)

def train(
        # Training parameters
        ways=10,
        shots=5,
        meta_lr=0.001,
        fast_lr=0.1,
        adapt_steps=5,
        meta_batch_size=32,
        iters=4000,
        cuda=True,
        seed=42,
        # Dataset parameters
        # Working conditions for different datasets, 
        # which should be in '0', '1', '2' or '3.
        train_domain=1,
        valid_domain=2,
        test_domain=3,
        # Data path
        data_dir_path='./data'
):
    # Set the Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set training device, using GPU if available
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device_count = torch.cuda.device_count()
        device = torch.device('cuda')
        logging.info('Training MAML with {} GPU(s).'.format(device_count))
    else:
        device = torch.device('cpu')
        logging.info('Training MAML with CPU.')
    
    # Create Datasets
    train_dataset = CWRU(train_domain,
                         data_dir_path,)
    # logging.info
    valid_dataset = CWRU(valid_domain,
                         data_dir_path,)
    
    test_dataset = CWRU(test_domain,
                        data_dir_path,)
    
    # Create Meta-Datasets
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    # Create Meta-Tasks
    train_transforms = [
        FusedNWaysKShots(train_dataset, n=ways, k=2*shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(
        train_dataset,
        task_transforms=train_transforms,
        num_tasks=400,
    )

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=ways, k=2*shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.Taskset(
        valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=100,
    )

    test_transforms = [
        FusedNWaysKShots(test_dataset, n=ways, k=2*shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=100,
    )

    # Create Model
    model = l2l.vision.models.CNN4(output_size=10)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = torch.optim.Adam(model.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    for iteration in range(1, iters+1):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0


        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()
            
            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        train_acc.append(meta_train_accuracy / meta_batch_size)
        valid_acc.append(meta_valid_accuracy / meta_batch_size)
        train_loss.append(meta_train_error / meta_batch_size)
        valid_loss.append(meta_valid_error / meta_batch_size)

        # Print some metrics
        # if (iteration % 10 == 0):
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        print('Meta Test Error', meta_test_error / meta_batch_size)
        print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

        # ========================= plot ==========================
        if (iteration % 10 == 0):
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.plot(train_acc, '-o', label="train acc")
            plt.plot(valid_acc, '-o', label="valid acc")
            plt.xlabel('Trainin iteration')
            plt.ylabel('Accuracy')
            plt.title("Accuracy Curve by Iteration")
            plt.legend()
            plt.subplot(122)
            plt.plot(train_loss, '-o', label="train loss")
            plt.plot(valid_loss, '-o', label="valid loss")
            plt.xlabel('Trainin iteration')
            plt.ylabel('Loss')
            plt.title("Loss Curve by Iteration")
            plt.legend()
            plt.suptitle("CWRU Bearing Fault Diagnosis {}way-{}shot".format(ways, shots))
            plt.savefig('./results/image_{}.png'.format(iteration))

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

if __name__ == '__main__':
    train()