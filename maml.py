from fault_datasets import CWRU, CWRU_FFT, HST, HST_FFT
from models import CNN1D
from utils import (
    print_logs,
    fast_adapt,
)

import logging
import torch
import random
import numpy as np
import learn2learn as l2l
import matplotlib.pyplot as plt

from torch import nn
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)

def train(args, experiment_title):
    """
    Train the MAML model on the specified dataset

    Args:
        args: parsed arguments
    """
    logging.info('Experiment: {}'.format(experiment_title))
    # Set the Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set training device, using GPU if available
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device_count = torch.cuda.device_count()
        device = torch.device('cuda')
        logging.info('Training MAML with {} GPU(s).'.format(device_count))
    else:
        device = torch.device('cpu')
        logging.info('Training MAML with CPU.')
    
    train_tasks, test_tasks = create_datasets(args)
    model, maml, opt, loss = create_model(args, device)

    train_model(args, model, maml, opt, loss, train_tasks, test_tasks, device, experiment_title)


def create_datasets(args):
    """
    Create the training, validation, and testing datasets
    
    Args:
        args: parsed arguments
    Returns:
        train_tasks: training tasks
        valid_tasks: validation tasks
        test_tasks: testing tasks
    """
    logging.info('Training domains: {}.'.format(args.train_domains))
    logging.info('Testing domain: {}.'.format(args.test_domain))
    train_datasets = []
    train_transforms = []
    train_tasks = []


    for i in range(len(args.train_domains)):
        if args.preprocess == 'FFT':
            if args.dataset == 'HST':
                train_datasets.append(HST_FFT(args.train_domains[i], 
                                              args.data_dir_path))
            else:
                train_datasets.append(CWRU_FFT(args.train_domains[i], 
                                               args.data_dir_path))
        else:
            if args.dataset == 'HST':
                train_datasets.append(HST(args.train_domains[i], 
                                           args.data_dir_path, 
                                           args.preprocess))
            else: 
                train_datasets.append(CWRU(args.train_domains[i], 
                                           args.data_dir_path, 
                                           args.preprocess)) 
        train_datasets[i] = l2l.data.MetaDataset(train_datasets[i])
        train_transforms.append([
            FusedNWaysKShots(train_datasets[i], n=args.ways, k=2*args.shots),
            LoadData(train_datasets[i]),
            RemapLabels(train_datasets[i]),
            ConsecutiveLabels(train_datasets[i]),
        ])
        train_tasks.append(l2l.data.Taskset(
            train_datasets[i],
            task_transforms=train_transforms[i],
            num_tasks=args.train_task_num,
        ))
    if args.preprocess == 'FFT':
        if args.dataset == 'HST':
            test_dataset = HST_FFT(args.test_domain, 
                                   args.data_dir_path)
        else:
            test_dataset = CWRU_FFT(args.test_domain, 
                                    args.data_dir_path)
    else:
        if args.dataset == 'HST':
            test_dataset = HST(args.test_domain, 
                               args.data_dir_path,
                               args.preprocess)
        else:
            test_dataset = CWRU(args.test_domain, 
                                args.data_dir_path,
                                args.preprocess)
    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=args.ways, k=2*args.shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=args.test_task_num,
    )

    return train_tasks, test_tasks


def create_model(args, device):
    """
    Create the MAML model, the MAML algorithm, the optimizer, and the loss function
    
    Args:
        args: parsed arguments
        device: device to run the model on
    Returns:
        model: the MAML model
        maml: the MAML algorithm
        opt: the optimizer
        loss: the loss function
    """
    output_size=10
    if args.dataset == 'HST':
        output_size=5
    if args.preprocess == 'FFT':
        model = CNN1D(output_size=output_size)
    else:
        model = l2l.vision.models.CNN4(output_size=output_size)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=args.first_order)
    opt = torch.optim.Adam(model.parameters(), args.meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    return model, maml, opt, loss


def train_model(args, model, maml, opt, loss, 
                train_tasks, test_tasks, 
                device, 
                experiment_title):
    train_acc_list = []
    train_err_list = []
    test_acc_list = []
    test_err_list = []

    # train_domains = args.train_domains.split(',')
    # train_domains = [int(i) for i in train_domains]

    for iteration in range(1, args.iters+1):
        opt.zero_grad()
        meta_train_err_sum = 0.0
        meta_train_acc_sum = 0.0
        meta_test_err_sum = 0.0
        meta_test_acc_sum = 0.0

        train_index = random.randint(0, len(args.train_domains)-1)

        for task in range(args.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks[train_index].sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            evaluation_error.backward()
            meta_train_err_sum += evaluation_error.item()
            meta_train_acc_sum += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               args.adapt_steps,
                                                               args.shots,
                                                               args.ways,
                                                               device)
            meta_test_err_sum += evaluation_error.item()
            meta_test_acc_sum += evaluation_accuracy.item()

        meta_train_acc = meta_train_acc_sum / args.meta_batch_size
        meta_train_err = meta_train_err_sum / args.meta_batch_size
        meta_test_err = meta_test_err_sum / args.meta_batch_size
        meta_test_acc = meta_test_acc_sum / args.meta_batch_size

        train_acc_list.append(meta_train_acc)
        test_acc_list.append(meta_test_acc)
        train_err_list.append(meta_train_err)
        test_err_list.append(meta_test_err)

        # Plot
        if args.plot and iteration % args.plot_step == 0:
            plot_metrics(args, 
                         iteration, 
                         train_acc_list, test_acc_list, 
                         train_err_list, test_err_list, 
                         experiment_title)

        # Save the model checkpoint
        if args.checkpoint and iteration % args.checkpoint_step == 0:
            torch.save(model.state_dict(), 
                       args.checkpoint_path + '/' +
                       experiment_title + 
                       '_{}.pt'.format(iteration))
        # Log some metrics
        if args.log:
            print_logs(iteration, meta_train_err, meta_train_acc, meta_test_err, meta_test_acc)

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
        opt.step()


def plot_metrics(args, 
                 iteration, 
                 train_acc, test_acc, 
                 train_loss, test_loss, 
                 experiment_title):
    if (iteration % args.plot_step == 0):
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.plot(train_acc, '-o', label="train acc")
        plt.plot(test_acc, '-o', label="test acc")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Curve by Iteration")
        plt.legend()
        plt.subplot(122)
        plt.plot(train_loss, '-o', label="train loss")
        plt.plot(test_loss, '-o', label="test loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title("Loss Curve by Iteration")
        plt.legend()
        # plt.suptitle("CWRU Bearing Fault Diagnosis {}way-{}shot".format(args.ways, args.shots))
        plt.savefig(args.plot_path + '/' + experiment_title + '_{}.png'.format(iteration))
        plt.show()


if __name__ == '__main__':
    train()