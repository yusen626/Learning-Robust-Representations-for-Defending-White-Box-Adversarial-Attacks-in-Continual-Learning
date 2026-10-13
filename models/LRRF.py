# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import torchattacks
import numpy as np
import torch
import torch.nn as nn
import copy
from backbone.HSIC_ import *
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *
from backbone.mmdloss import *

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)
metric = MaximumMeanDiscrepancy()
metric.attach(default_evaluator, "mmd")

class DeFenceAdvanced5(ContinualModel):
    NAME = 'derfenceadvanced5'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                   ' Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--r_alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--r_alpha2', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--distance', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--buffer_type', type=str, default='ours', choices=['clean', 'adversarial', 'mixed', 'ours'], 
                        help='Type of memory buffer for ablation study')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DeFenceAdvanced5, self).__init__(backbone, loss, args, transform)
        if not hasattr(self.net, 'forward'):
            raise ValueError("backbone must be a model instance with 'forward' method")
        self.buffer = Buffer(self.args.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.is_vit = 'VisionTransformer' in str(type(backbone))
        if not hasattr(self.net, 'forward'):
            raise ValueError("ViT backbone must have 'forward' method")
        self.adversary = torchattacks.PGD(self.net, eps=8 / 255, alpha=2 / 255, steps=5, random_start=True)
        self.mse_loss = nn.MSELoss()
        self.mmdLoss = MMSD()
        self.mmdLoss.to(self.device)
        self.buffer_type = args.buffer_type  # Store the selected memory buffer type.

    def CalculateDistance(self, a1, a2):
        # Additional metrics: cosine alignment (option 3) and KL on features (option 4).
        if self.args.distance == 0:  # MSE
            return self.mse_loss(a1, a2)
        elif self.args.distance == 1:  # HSIC
            return HSIC(a1, a2)
        elif self.args.distance == 2:  # MMD
            return self.mmdLoss(a1, a2)
        elif self.args.distance == 3:  # Cosine alignment 
            cos_sim = F.cosine_similarity(a1, a2, dim=1)
            return 1.0 - torch.mean(cos_sim)
        elif self.args.distance == 4:  # KL on features 
            p = F.softmax(a1, dim=1)
            q = F.softmax(a2, dim=1)
            # KL(p||q)
            kl = F.kl_div(F.log_softmax(a1, dim=1), q, reduction='none').sum(dim=1)
            return torch.mean(kl)
        else:
            raise ValueError(f"Invalid distance metric: {self.args.distance}")

    def end_task(self, dataset):
        self.Teacher = copy.deepcopy(self.net)
        self.Teacher = self.Teacher.to(self.device)
        for param in self.Teacher.parameters():
            param.requires_grad = False

    def get_cls_features(self, x, net1=None):
        # Validate that x is not None.
        if x is None:
            raise ValueError("Input x is None in get_cls_features")
        
        if net1 is not None:
            features = net1(x)
        else:
            features = self.net(x)
        
        if self.is_vit:
            if isinstance(features, tuple):
                features = features[1]
            if len(features.shape) == 3:
                return features[:, 0, :]
            elif len(features.shape) == 2:
                return features
            else:
                # Return the original features if their shape does not match.
                return features
        else:
            _, _, _, _, features = self.net.ReturnAllFeatures(x)
            return features

    def Calculate_DistanceBetwenTwoSamples(self, a1, a2):
        # Validate that a1 and a2 are not None.
        if a1 is None or a2 is None:
            raise ValueError("a1 or a2 is None in Calculate_DistanceBetwenTwoSamples")
        
        features1 = self.get_cls_features(a1)
        features2 = self.get_cls_features(a2)
        return self.CalculateDistance(features1, features2)

    def CalculateDistanceBetweenTwoModels(self, a1, net1, net2):
        features1 = self.get_cls_features(a1, net1)
        features2 = self.get_cls_features(a1, net2)
        return self.CalculateDistance(features1, features2)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        # Validate that inputs and labels are not None.
        if inputs is None or labels is None:
            raise ValueError("inputs or labels is None in observe")
        
        self.opt.zero_grad()
        tot_loss = 0
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        if self.current_task == 0:
            # Ensure that inputs is not None.
            if inputs is None:
                raise ValueError("inputs is None in current_task == 0")
            
            newInput, newlabels = inputs.cuda(), labels.cuda()
            # Ensure that newInput is not None.
            if newInput is None:
                raise ValueError("newInput is None after inputs.cuda()")
            
            self.net.eval()
            aSamples = self.adversary(newInput, newlabels)
            self.net.train()
            
            # Ensure that aSamples is not None.
            if aSamples is None:
                raise ValueError("aSamples is None after self.adversary()")
            
            adversaral_outputs = self.net(aSamples)
            loss2 = self.loss(adversaral_outputs, labels)
            loss2.backward()
            tot_loss += loss2
            
            sumd1 = self.args.r_alpha * self.Calculate_DistanceBetwenTwoSamples(newInput, aSamples)
            sumd1.backward()
            tot_loss += sumd1.item()
            
            if not self.buffer.is_empty():
                buf_inputs, _, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                
                # Ensure that buf_inputs is not None.
                if buf_inputs is None:
                    raise ValueError("buf_inputs is None after buffer.get_data()")
                
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                
                num_classes = np.shape(buf_logits)[1]
                buf_labels = torch.argmax(buf_logits, 1)
                
                if self.current_task == 0:
                    buf_inputs, buf_labels = buf_inputs.cuda(), buf_labels.cuda()
                    
                    # Ensure that buf_inputs is not None.
                    if buf_inputs is None:
                        raise ValueError("buf_inputs is None after buffer.get_data()")
                    
                    self.net.eval()
                    aSamples = self.adversary(buf_inputs, buf_labels)
                    self.net.train()
                    
                    aOutputs = self.net(aSamples)
                    loss_mse2 = self.args.alpha * F.mse_loss(aOutputs, buf_logits)
                    loss_mse2.backward()
                    
                    sumd = self.args.r_alpha * self.Calculate_DistanceBetwenTwoSamples(buf_inputs, aSamples)
                    sumd.backward()
                    tot_loss += loss_mse.item() + loss_mse2.item() + sumd.item()
                else:
                    loss_ce_teacher = self.args.r_alpha2 * self.CalculateDistanceBetweenTwoModels(
                        inputs, self.net, self.Teacher)
                    loss_ce_teacher.backward()
                    tot_loss += loss_ce_teacher.item()
                    
                    buf_labels = torch.argmax(buf_logits, 1)
                    newInput, newlabels = buf_inputs.cuda(), buf_labels.cuda()
                    
                    self.net.eval()
                    aSamples = self.adversary(newInput, newlabels)
                    self.net.train()
                    
                    sumd = self.args.r_alpha * self.Calculate_DistanceBetwenTwoSamples(buf_inputs, aSamples)
                    sumd.backward()
                    tot_loss += sumd
        else:
            # Handle subsequent tasks.
            loss_ce_teacher = self.args.r_alpha2 * self.CalculateDistanceBetweenTwoModels(
                inputs, self.net, self.Teacher)
            loss_ce_teacher.backward()
            tot_loss += loss_ce_teacher.item()
            
            if not self.buffer.is_empty():
                buf_inputs, _, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                
                # Ensure that buf_inputs is not None.
                if buf_inputs is None:
                    raise ValueError("buf_inputs is None after buffer.get_data()")
                
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                
                num_classes = np.shape(buf_logits)[1]
                buf_labels = torch.argmax(buf_logits, 1)
                
                # Ensure that buf_inputs is not None.
                if buf_inputs is None:
                    raise ValueError("buf_inputs is None after buffer.get_data()")
                
                newInput, newlabels = buf_inputs.cuda(), buf_labels.cuda()
                
                self.net.eval()
                aSamples = self.adversary(newInput, newlabels)
                self.net.train()
                
                sumd = self.args.r_alpha * self.Calculate_DistanceBetwenTwoSamples(buf_inputs, aSamples)
                sumd.backward()
                tot_loss += sumd

        self.opt.step()
        
        if self.current_task == 0:
            # Store samples according to buffer_type.
            if self.buffer_type == 'clean':
                self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
            elif self.buffer_type == 'adversarial':
                self.buffer.add_data(examples=aSamples, labels=labels, logits=outputs.data)
            elif self.buffer_type == 'mixed':
                # Store both clean and adversarial samples.
                self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
                self.buffer.add_data(examples=aSamples, labels=labels, logits=outputs.data)
            elif self.buffer_type == 'ours':
                self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
            else:
                raise ValueError(f"Invalid buffer_type: {self.buffer_type}")
        
        return tot_loss
