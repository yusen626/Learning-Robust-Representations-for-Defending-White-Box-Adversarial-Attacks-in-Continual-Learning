# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel

from utils import random_id
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.status import ProgressBar
from time import time
from tqdm import tqdm
import torchattacks
from robustbench.utils import load_model, clean_accuracy
from backbone.ResNet18 import *

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate_ensemble(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.modelArr[0].training #model.net.training

    for c1 in range(np.shape(model.modelArr)[0]):
        #model.net.eval()
        model.modelArr[c1].eval()

    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    expertIndex = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model.modelArr[k](inputs) #model(inputs, k)
            else:
                outputs = model.modelArr[k](inputs) #model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        expertIndex = expertIndex +1

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    for c1 in range(np.shape(model.modelArr)[0]):
        #model.net.eval()
        model.modelArr[c1].train(status)
    #model.net.train(status)
    return accs, accs_mask_classes


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if model.NAME == 'dynamicdefense':
                    outputs = model.myPrediction(inputs,k)
                elif model.NAME == 'dynamicdefense2' :
                    outputs = model.myPrediction(inputs, k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def evaluate_attacks(k,model: ContinualModel, dataset: ContinualDataset, last=False,attackArr=[]) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]

    train_loader, test_loader = dataset.getAlldataLoaders()
    test_iter = iter(test_loader)
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    i = 0

    if k == 0:
        totalTestingX = []
        totalTestingY = []
        '''
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break

            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)

            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

        acc = correct/total
        '''
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break

            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            sInputs = inputs

            if np.shape(totalTestingX)[0] == 0:
                totalTestingX = sInputs
                totalTestingY = labels
            else:
                totalTestingX = torch.cat((totalTestingX,sInputs),0)
                totalTestingY = torch.cat((totalTestingY,labels),0)

        if model.NAME == 'dynamicattackdem':
            print("this is dynamicattackdem")
            model.SetCurrentExpertByIndex(k)

        acc = clean_accuracy(model, totalTestingX, totalTestingY)
    else:
        myK = k-1
        totalTestingX = []
        totalTestingY = []
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break

            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            sInputs = attackArr[myK](inputs,labels)

            if np.shape(totalTestingX)[0] == 0:
                totalTestingX = sInputs
                totalTestingY = labels
            else:
                totalTestingX = torch.cat((totalTestingX,sInputs),0)
                totalTestingY = torch.cat((totalTestingY,labels),0)

        if model.NAME == 'dynamicattackdem':
            #print("this is dynamicattackdem")
            model.SetCurrentExpertByIndex(k)
        acc = clean_accuracy(model, totalTestingX, totalTestingY)

    return acc

def evaluate_defense(model: ContinualModel, dataset: ContinualDataset, last=False,attackName="") -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        attackName: the attack to use, either 'FGSM' or 'PGD'

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    if attackName not in ('FGSM', 'PGD'):
        raise ValueError(f"Unsupported attack {attackName!r}. Choose 'FGSM' or 'PGD'.")

    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    #atk = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=4)

    if attackName == 'FGSM':
        print("This is FGSM")
        atk = torchattacks.FGSM(model, eps=8 / 255)
    elif attackName == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)

    totalTest = []
    totalTestLabel = []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            #Produce the adversarial samples
            adv_images = atk(inputs, labels)
            if np.shape(totalTest)[0] == 0:
                totalTest = adv_images
                totalTestLabel = labels
            else:
                totalTest = torch.cat([totalTest,adv_images],0)
                totalTestLabel = torch.cat([totalTestLabel,labels],0)

            '''
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(adv_images, k)
            else:
                if model.NAME == 'hybridmixture':
                    outputs = model.myPrediction(inputs,k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
            '''

        '''
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        '''

    acc = clean_accuracy(model, totalTest, totalTestLabel)
    model.net.train(status)
    return acc
    #return accs, accs_mask_classes


def evaluate_defense_ensemble(model: ContinualModel, dataset: ContinualDataset, last=False,attackName="") -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        attackName: the attack to use, either 'FGSM' or 'PGD'

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    if attackName not in ('FGSM', 'PGD'):
        raise ValueError(f"Unsupported attack {attackName!r}. Choose 'FGSM' or 'PGD'.")

    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    #atk = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=4)

    if attackName == 'FGSM':
        print("This is FGSM")
        atk = torchattacks.FGSM(model, eps=8 / 255)
    elif attackName == 'PGD':
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=False)

    accArr = []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        totalTest = []
        totalTestLabel = []

        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            #Produce the adversarial samples
            adv_images = atk(inputs, labels)
            if np.shape(totalTest)[0] == 0:
                totalTest = adv_images
                totalTestLabel = labels
            else:
                totalTest = torch.cat([totalTest,adv_images],0)
                totalTestLabel = torch.cat([totalTestLabel,labels],0)

            '''
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(adv_images, k)
            else:
                if model.NAME == 'hybridmixture':
                    outputs = model.myPrediction(inputs,k)
                else:
                    outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
            '''
        model.SetCurrentExpertByIndex(k)
        acc1 = clean_accuracy(model, totalTest, totalTestLabel)
        accArr.append(acc1)
        '''
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        '''

    acc = np.mean(accArr)
    model.net.train(status)
    return acc
    #return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = random_id(5)
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()


def evaluateAccuracy(model, dataset, k, attackArr):
    myarr = []
    
    for task_id in range(k + 1):
        acc = evaluate_attacks(task_id, model, dataset, attackArr=attackArr)
        myarr.append(acc)
    print("all tasks")
    print(myarr)
    print("average accuracy")
    print(np.mean(myarr))


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger.load(csvdump)

        print('Checkpoint Loaded!')
    
    if args.buffer_type not in ['clean', 'adversarial', 'mixed', 'ours']:
        raise ValueError(f"Invalid buffer_type: {args.buffer_type}")

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)
    numTask = 3

    if args.enable_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()

    currentNet = 0

    #Train a single classifier
    train_loader, test_loader = dataset.getAlldataLoaders()

    nclasses = dataset.N_CLASSES
    nf = 64

    mynet = MyResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
    totalEpoch = model.args.n_epochs

    mynet.to(model.device)

    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

    totalEpoch = 1
    for c in range(totalEpoch):
        train_iter = iter(train_loader)
        previous_time = time()
        data_len = len(train_loader)
        mininterval = 0.5 if data_len is not None and data_len > 1000 else 0.1
        pbar = tqdm(train_iter, total=data_len,
                    desc=f"Epoch {c + 1}",
                    disable=args.non_verbose, mininterval=mininterval, ncols=150)
        while True:
            try:
                data = next(train_iter)
            except StopIteration:
                isFirst = False
                break
            if args.debug_mode and i > model.get_debug_iters():
                break
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, not_aug_inputs, logits = data
                inputs = inputs.to(model.device)
                labels = labels.to(model.device, dtype=torch.long)
                not_aug_inputs = not_aug_inputs.to(model.device)
                logits = logits.to(model.device)
                # loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                # if epoch == 0:
                #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
            else:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                not_aug_inputs = not_aug_inputs.to(model.device)
                # loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

            optimizer.zero_grad()
            outputs = mynet(inputs)
            loss1 = F.cross_entropy(outputs, labels)
            loss1.backward()
            optimizer.step()

            time_diff = time() - previous_time
            previous_time = time()
            bar_log = {'loss': loss1}
            if data_len:
                ep_h = 3600 / (data_len * time_diff)
                bar_log['ep/h'] = ep_h
            pbar.set_postfix(bar_log, refresh=False)
            pbar.update()

        pbar.close()
    # End the training
    # Create a number of attacks

    atk1 = torchattacks.FGSM(model.net, eps=8 / 255)
    atk2 = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=10, random_start=False)
    attackArr = []
    attackArr.append(atk1)
    attackArr.append(atk2)
    currentNet = mynet

    numTask = np.shape(attackArr)[0] + 1

    start = time()
    for t in range(numTask):
        model.net.train()
        model.meta_begin_task(dataset)

        if t == 0:  # if task ==1
            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)

                previous_time = time()
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)

                mininterval = 0.5 if data_len is not None and data_len > 1000 else 0.1
                pbar = tqdm(train_iter, total=data_len,
                            desc=f"Task {t + 1} - Epoch {epoch + 1}",
                            disable=args.non_verbose, mininterval=mininterval, ncols=150)
                
                i = 0
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        isFirst = False
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                        #if epoch == 0:
                        #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
                    
                    time_diff = time() - previous_time
                    previous_time = time()
                    bar_log = {}
                    if data_len:
                        ep_h = 3600 / (data_len * time_diff)
                        bar_log['ep/h'] = ep_h
                    pbar.set_postfix(bar_log, refresh=False)
                    pbar.update()
                  
                pbar.close()

        else:
            totalTrainX = []
            totalTrainY = []

            #Produce the adversarial samples
            #atk1 = torchattacks.FGSM(model.net, eps=8 / 255)
            #atk2 = torchattacks.PGD(model.net, eps=8 / 255, alpha=2 / 255, steps=10, random_start=False)
            #attackArr = []
            #attackArr.append(atk1)
            #attackArr.append(atk2)

            train_iter = iter(train_loader)

            previous_time = time()
            data_len = len(train_loader)
            mininterval = 0.5 if data_len is not None and data_len > 1000 else 0.1
            pbar = tqdm(train_iter, total=data_len,
                        desc=f"Task {t + 1}",
                        disable=args.non_verbose, mininterval=mininterval, ncols=150)
            while True:
                try:
                    data = next(train_iter)
                except StopIteration:
                    isFirst = False
                    break
                if args.debug_mode and i > model.get_debug_iters():
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    # loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                    # if epoch == 0:
                    #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    # loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

                myT = t - 1
                atk = attackArr[myT]
                mynet.eval()
                model.net.eval()

                sInputs = atk(inputs, labels)
                # Convert the data to numpy and store it in the CPU
                if np.shape(totalTrainX)[0] == 0:
                    totalTrainX = sInputs.cpu().numpy()
                    totalTrainY = labels.cpu().numpy()
                else:
                    totalTrainX = np.concatenate((totalTrainX, sInputs.cpu().numpy()), axis=0)
                    totalTrainY = np.concatenate((totalTrainY, labels.cpu().numpy()), axis=0)

                time_diff = time() - previous_time
                previous_time = time()
                bar_log = {}
                if data_len:
                    ep_h = 3600 / (data_len * time_diff)
                    bar_log['ep/h'] = ep_h
                pbar.set_postfix(bar_log, refresh=False)
                pbar.update()

                model.net.train()

            pbar.close()
            #perform the training process
            miniBatchSize = 32  
            for epoch in range(model.args.n_epochs):
                for j in range(0, totalTrainX.shape[0], miniBatchSize):
                    # load data from numpy
                    batch_data = totalTrainX[j:j+miniBatchSize]
                    batch_labels = totalTrainY[j:j+miniBatchSize]
                    
                    # convert numpy to tensor && move data to GPU
                    inputs = torch.from_numpy(batch_data).to(model.device, dtype=torch.float)
                    labels = torch.from_numpy(batch_labels).to(model.device, dtype=torch.long)
                    
                    # 224x224 -> 32x32 
                    not_aug_inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
                
                # Clear the GPU cache at the end of each epoch
                torch.cuda.empty_cache()

        # model.end_task2(dataset,train_loader)
        # return
        # model.end_task2(dataset, train_loader,selectedMaxSize)
        model.meta_end_task(dataset)

        end = time()
        TrainingTime = end - start
        print("Total training time")
        print(TrainingTime)

        evaluateAccuracy(model, dataset, t, attackArr)

        '''
        if t == 0:
            acc = evaluate_attacks(t,model,dataset,attackArr=attackArr)
            print("Current acc:")
            print(acc)
        '''
        #accs = evaluate(model, dataset)
        #results.append(accs[0])
        #results_mask_classes.append(accs[1])

    #Calculate the final accuracy
    evaluateAccuracy(model, dataset, numTask-1, attackArr)

    '''
    myarr = []
    acc1 = evaluate_attacks(0,model,dataset,attackArr=attackArr)
    myarr.append(acc1)
    for j in range(np.shape(attackArr)[0]):
        acc = evaluate_attacks(j,model,dataset,attackArr=attackArr)
        myarr.append(acc)

    print("all tasks")
    print(myarr)

    print("average accuracy")
    print(np.mean(myarr))
    '''

def train_attack(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger.load(csvdump)

        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)
    numTask = 3

    if args.enable_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()

    currentNet = 0

    start = time()
    for t in range(start_task, end_task):
        model.net.train()
        train_loader, test_loader = dataset.getAlldataLoaders()
        model.meta_begin_task(dataset)

        if t == 0:#if task ==1
            #Train a classifier network
            nclasses = dataset.N_CLASSES
            nf = 64
            mynet = MyResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
            totalEpoch = 100

            train_iter = iter(train_loader)

            optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

            for c in range(totalEpoch):
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        isFirst = False
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        #loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                        #if epoch == 0:
                        #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        #loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

                    optimizer.zero_grad()
                    outputs = mynet(inputs)
                    loss1 = F.cross_entropy(outputs, labels)
                    loss1.backward()
                    optimizer.step()
            currentNet = mynet
            #End the training

        else:
            totalTrainX = []
            totalTrainY = []

            while True:
                try:
                    data = next(train_iter)
                except StopIteration:
                    isFirst = False
                    break
                if args.debug_mode and i > model.get_debug_iters():
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    # loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                    # if epoch == 0:
                    #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    # loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

                if np.shape(totalTrainX)[0] == 0:
                    totalTrainX = inputs
                    totalTrainY = labels
                else:
                    totalTrainX = torch.cat((totalTrainX,inputs),0)
                    totalTrainY = torch.cat((totalTrainY,labels),0)

            #Create the new adversarial samples as the training dataset
            if t == 1:
                atk = torchattacks.FGSM(currentNet, eps=8 / 255)
            else:
                atk = torchattacks.PGD(currentNet, eps=8 / 255, alpha=2 / 255, steps=10, random_start=False)

            adv_images = atk(totalTrainX, totalTrainY)

        myCount = len(train_loader) * 32
        print(myCount)
        return

        #model.end_task2(dataset,train_loader)
        #return

        if not args.inference_only:
            if t and args.enable_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t - 1] = results[t - 1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

            scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler

            #Add data sample
            maxSize = model.args.buffer_size
            longTermSize = int(maxSize / 2)
            maxNofTasks = dataset.N_TASKS

            selectedMaxSize = int(longTermSize / maxNofTasks)
            dataX, dataY, dataLogit = [], [], []
            #end data

            isFirst = True
            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)
                i = 0
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        isFirst = False
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                        #if epoch == 0:
                        #    model.TSFramework.AddSupervisedDataBatch(inputs, labels,logits)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)

                        #Add data
                        if epoch == 0:
                            with torch.no_grad():
                                outputs = model.net(inputs)

                            labels2 = labels
                            inputs2 = inputs
                            not_aug_inputs2 = not_aug_inputs
                            outputs2 = outputs

                            #inputs2 = inputs2.to(torch.device("cpu"))
                            not_aug_inputs2 = not_aug_inputs2.to(torch.device("cpu"))
                            outputs2 = outputs2.to(torch.device("cpu"))

                            if np.shape(dataX)[0] == 0:
                                dataX = not_aug_inputs2
                                dataY = labels2
                                dataLogit = outputs2
                            else:
                                # print(dataX.size())
                                dataX = torch.cat((dataX, not_aug_inputs2), 0)
                                dataY = torch.cat((dataY, labels2), 0)
                                dataLogit = torch.cat((dataLogit, outputs2), 0)
                        #End data

                    assert not math.isnan(loss)
                    #progress_bar.prog(i, data_len, epoch, t, loss)
                    i += 1
                    #return
                if scheduler is not None:
                    scheduler.step()

                if args.eval_epochs is not None and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs - 1:
                    epoch_accs = evaluate(model, dataset)

                    log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

            dataX = torch.reshape(dataX, (-1, 3, 32, 32))
            dataY = torch.reshape(dataY, (np.shape(dataY)[0], -1))
            dataLogit = torch.reshape(dataLogit, (np.shape(dataLogit)[0], -1))

            if model.NAME == 'pnn':
                model.processData(dataX,dataY,dataLogit,selectedMaxSize)

        #model.end_task2(dataset, train_loader,selectedMaxSize)
        model.meta_end_task(dataset)

        end = time()
        TrainingTime = end-start
        print("Total training time")
        print(TrainingTime)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        log_accs(args, logger, accs, t, dataset.SETTING)

        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [results, results_mask_classes, logger.dump()],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if 'buffer_size' in model.args:
                save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
            torch.save(save_obj, checkpoint_name)

    #Calculate the number of parameters
    total = sum([param.nelement() for param in model.net.parameters()])
    print("The total number of parameters for the model")
    print(total)

    if model.NAME == 'dynamicdefense':
        # Perform the adversarial evaluation
        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='FGSM')
        print("FGSM")
        print(myAccsArr)

        myAccsArr = evaluate_defense_ensemble(model, dataset, attackName='PGD')
        print("PGD")
        print(myAccsArr)

    else:
        #Perform the adversarial evaluation
        myAccsArr = evaluate_defense(model, dataset,attackName='FGSM')
        print("FGSM")
        print(myAccsArr)

        myAccsArr = evaluate_defense(model, dataset,attackName='PGD')
        print("PGD")
        print(myAccsArr)

    if args.validation:
        del dataset
        args.validation = None

        final_dataset = get_dataset(args)
        for _ in range(final_dataset.N_TASKS):
            final_dataset.get_data_loaders()
        accs = evaluate(model, final_dataset)
        log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

    if not args.disable_log and args.enable_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
