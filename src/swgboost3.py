#==========================================
# Header
#==========================================
# Copyright (c) Takuo Matsubara
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.



#==========================================
# Import Library
#==========================================
import numpy as np
import scipy.stats as stat
import scipy.optimize as soptim
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
from torch.func import grad, jacrev, vmap
from typing import List, Optional, Callable

import time
import argparse
from src.GrowNet_Regression.data.sparseloader import DataLoader
from src.GrowNet_Regression.data.data import LibSVMData, LibCSVData, LibSVMRegData
from src.GrowNet_Regression.data.sparse_data import LibSVMDataSp
from src.GrowNet_Regression.models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from src.GrowNet_Regression.models.dynamic_net import DynamicNet, ForwardType
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim import SGD, Adam
from copy import deepcopy
from src.data import TensorDataset
from sklearn.model_selection import train_test_split




#==========================================
# Main Classes
#==========================================
class SWGBoost3(BaseEstimator):
    def __init__(
            self, grad_logp, hess_logp, learner_class,
            learner_param: dict = None,
            learning_rate: float = 0.01,
            n_estimators: int = 500,
            subsample: float = 1.0,
            n_particles: int = 10,
            d_particles: int = 1,
            bandwidth: float = 1.0,
            random_state: int = 0,
            init_iter: float = 5000,
            init_lr: float = 0.1,
            init_locs: np.ndarray = None,
        ):
        
        self.grad_logp = grad_logp
        self.hess_logp = hess_logp
        self.grad_logp_vmap = vmap(vmap(self.grad_logp, in_dims=(0,0)), in_dims=(0,0))
        self.hess_logp_vmap = vmap(vmap(self.hess_logp, in_dims=(0,0)), in_dims=(0,0))
        
        self.learner_class = learner_class
        self.learner_param = learner_param if learner_param is not None else {}
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_particles = n_particles
        self.d_particles = d_particles
        self.bandwidth = bandwidth
        self.subsample = subsample
        
        self.base0 = np.zeros((n_particles, d_particles))
        self.bases = []
        self.rates = []
        self.rng = np.random.default_rng(random_state)

        self.init_iter = init_iter
        self.init_lr = init_lr
        self.init_locs = init_locs if init_locs is not None else self.rng.normal(0, 1, size=(n_particles, d_particles))


    def init_opt(self, feat_d, n_particles, d_particles):
        parser = argparse.ArgumentParser()
        parser.add_argument('--feat_d', default=feat_d, type=int)
        parser.add_argument('--hidden_d', default=10, type=int)
        parser.add_argument('--boost_rate', default=0.1, type=float)
        parser.add_argument('--lr', default=0.1, type=float)
        parser.add_argument('--num_nets', default=100, type=int)
        parser.add_argument('--data', default='test', type=str)
        parser.add_argument('--tr', default='None', type=str)
        parser.add_argument('--te', default='None', type=str)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--epochs_per_stage', default=100, type=int)
        parser.add_argument('--correct_epoch', default=100, type=int)
        parser.add_argument('--L2', default=1, type=float)
        parser.add_argument('--sparse', action='store_true')
        parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--out_f', default='./model.pt', type=str)
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--d_particles', default=d_particles, type=int)
        parser.add_argument("--n_particles", type=int, default=n_particles)

        opt = parser.parse_args()

        if not opt.cuda:
            torch.set_num_threads(16)

        return opt


    def get_optim(self, params, lr, weight_decay):
        optimizer = Adam(params, lr, weight_decay=weight_decay)
        #optimizer = SGD(params, lr, weight_decay=weight_decay)
        return optimizer
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # P = np.zeros((X.shape[0], self.n_particles, self.d_particles))
        # P += self.base0
        # for ith in range(len(self.bases)):
        #     P += self.learning_rate * self.rates[ith] * self._reshape_forward( self.bases[ith].predict(X) )
        X = torch.as_tensor(X, dtype=torch.float32)
        _, out = self.bases.forward_grad(X)
        out = out.detach().numpy()
        return out
    
    
    def predict_eachitr(self, X: np.ndarray) -> np.ndarray:
        # P = np.zeros((self.n_estimators, X.shape[0], self.n_particles, self.d_particles))
        # P[0] = self.base0 + self.learning_rate * self.rates[0] * self._reshape_forward( self.bases[0].predict(X) )
        # for ith in range(1, len(self.bases)):
        #     P[ith] = P[ith-1] + self.learning_rate * self.rates[ith] * self._reshape_forward( self.bases[ith].predict(X) )

        X = torch.as_tensor(X, dtype=torch.float32)
        P = np.zeros((self.opt.num_nets, X.shape[0], self.n_particles, self.d_particles))
        P[0] = self.bases.c0.expand(X.shape[0], self.bases.c0.shape[0], self.bases.c0.shape[1]).cpu().numpy()
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for ith, m in enumerate(self.bases.models):
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(X, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(X, middle_feat_cum)
                    prediction += pred
                if ith>0:
                    P[ith] = P[ith-1] + (self.bases.boost_rate * prediction.reshape(X.shape[0], self.bases.c0.shape[0], self.bases.c0.shape[1])).cpu().numpy()
                else:
                    P[ith] = P[ith] + (self.bases.boost_rate * prediction.reshape(X.shape[0], self.bases.c0.shape[0], self.bases.c0.shape[1])).cpu().numpy()
        return P


    # prepare the dataset
    def get_data(self, X=None, Y=None):
        if self.opt.data in ['ca_housing', 'ailerons', 'YearPredictionMSD', 'slice_localization']:
            train = LibSVMRegData(self.opt.tr, self.opt.feat_d, self.opt.normalization)
            test = LibSVMRegData(self.opt.te, self.opt.feat_d, self.opt.normalization)
            val = []
            if self.opt.cv:
                val = deepcopy(train)
                print('Creating Validation set! \n')
                indices = list(range(len(train)))
                cut = int(len(train)*0.95)
                np.random.shuffle(indices)
                train_idx = indices[:cut]
                val_idx = indices[cut:]

                train.feat = train.feat[train_idx]
                train.label = train.label[train_idx]
                val.feat = val.feat[val_idx]
                val.label = val.label[val_idx]
        elif X is not None and Y is not None:
            Xt, Xv, yt, yv = train_test_split(X, Y, test_size=0.2)
            train = TensorDataset(Xt, yt)
            test = TensorDataset(Xv, yv)
            val = []
            if self.opt.cv:
                val = deepcopy(train)
                print('Creating Validation set! \n')
                indices = list(range(len(train)))
                cut = int(len(train)*0.95)
                np.random.shuffle(indices)
                train_idx = indices[:cut]
                val_idx = indices[cut:]

                train.feat = train.feat[train_idx]
                train.label = train.label[train_idx]
                val.feat = val.feat[val_idx]
                val.label = val.label[val_idx]
        else:
            pass

        if self.opt.normalization:
            scaler = StandardScaler()
            scaler.fit(train.feat)
            train.feat = scaler.transform(train.feat)
            test.feat = scaler.transform(test.feat)
            if self.opt.cv:
                val.feat = scaler.transform(val.feat)
        print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
        return train, test, val


    def root_mse(self, net_ensemble, loader):
        loss = 0
        total = 0

        for x, y in loader:
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32)
            if self.opt.cuda:
                x = x.cuda()

            with torch.no_grad():
                _, out = net_ensemble.forward(x)
            y = y.cpu().numpy()
            y = y.reshape(y.size, 1)
            out = out.cpu().numpy()
            out = np.mean(out[:, :, 0], axis=1, keepdims=True)
            out = out.reshape(len(y), 1)
            loss += mean_squared_error(y, out) * len(y)
            total += len(y)
        return np.sqrt(loss / total)
    
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.opt = self.init_opt(X.shape[1], self.n_particles, self.d_particles)
        train, test, val = self.get_data(X, Y)
        N = len(train)
        print(self.opt.data + ' training and test datasets are loaded!')
        train_loader = DataLoader(train, self.opt.batch_size, shuffle=True, drop_last=False, num_workers=0)
        test_loader = DataLoader(test, self.opt.batch_size, shuffle=False, drop_last=False, num_workers=0)
        if self.opt.cv:
            val_loader = DataLoader(val, self.opt.batch_size, shuffle=True, drop_last=False, num_workers=0)
        best_rmse = pow(10, 6)
        val_rmse = best_rmse
        best_stage = self.opt.num_nets-1
        # c0 = np.mean(train.label)  #init_gbnn(train)
        # self.base0 = self.compute_init_base(Y)
        c0 = train.label.mean(dim=0)
        c0 = c0.expand(self.n_particles, self.d_particles)
        self.bases = DynamicNet(c0, self.opt.boost_rate)
        loss_f1 = nn.MSELoss()
        loss_models = torch.zeros((self.opt.num_nets, 3))
        
        for ith in range(self.opt.num_nets):
            
            if self.subsample == 1.0:
                t0 = time.time()
                model = MLP_2HL.get_model(ith, self.opt)  # Initialize the model_k: f_k(x), multilayer perception v2
                if self.opt.cuda:
                    model.cuda()

                optimizer = self.get_optim(model.parameters(), self.opt.lr, self.opt.L2)
                self.bases.to_train() # Set the models in ensemble net to train mode
                stage_mdlloss = []
                for epoch in range(self.opt.epochs_per_stage):
                    for i, (x, y) in enumerate(train_loader):
                        x = torch.as_tensor(x, dtype=torch.float32)
                        y = torch.as_tensor(y, dtype=torch.float32)

                        if self.opt.cuda:
                            x = x.cuda()
                            y = torch.as_tensor(y, dtype=torch.float32).cuda().view(-1, 1)
                        middle_feat, out = self.bases.forward(x)
                        if self.opt.cuda:
                            out = torch.as_tensor(out, dtype=torch.float32).cuda()
                        else:
                            out = torch.as_tensor(out, dtype=torch.float32)
                        G = self.gradient(out, y)
                        if self.opt.cuda:
                            G = torch.as_tensor(G, dtype=torch.float32).cuda()
                        else:
                            G = torch.as_tensor(G, dtype=torch.float32)
                        # grad_direction = -(out-y)

                        _, out = model(x, middle_feat)
                        if self.opt.cuda:
                            out = torch.as_tensor(out, dtype=torch.float32).cuda()
                        else:
                            out = torch.as_tensor(out, dtype=torch.float32)

                        loss = loss_f1(self.bases.boost_rate*out.squeeze(), self._reshape_backward(G).squeeze())  # T
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        stage_mdlloss.append(loss.item()*len(y))

                self.bases.add(model)
                sml = np.sqrt(np.sum(stage_mdlloss)/N)

                lr_scaler = 3
                # fully-corrective step
                stage_loss = []
                if ith > 0:
                    # Adjusting corrective step learning rate
                    if ith % 15 == 0:
                        #lr_scaler *= 2
                        self.opt.lr /= 2
                        self.opt.L2 /= 2
                    optimizer = self.get_optim(self.bases.parameters(), self.opt.lr / lr_scaler, self.opt.L2)
                    for _ in range(self.opt.correct_epoch):
                        stage_loss = []
                        for i, (x, y) in enumerate(train_loader):
                            x = torch.as_tensor(x, dtype=torch.float32)
                            y = torch.as_tensor(y, dtype=torch.float32)
                            if self.opt.cuda:
                                x, y = x.cuda(), y.cuda().view(-1, 1)
                            _, out = self.bases.forward_grad(x)
                            if self.opt.cuda:
                                out = torch.as_tensor(out, dtype=torch.float32).cuda()
                            else:
                                out = torch.as_tensor(out, dtype=torch.float32)

                            # temp = self._reshape_forward(out.cpu().detach())
                            # temp = torch.from_numpy(np.mean(temp[:, :, 0].numpy(), axis=1, keepdims=True)).squeeze()
                            if len(y.shape) == 2:
                                y = y.unsqueeze(2)
                            temp = y.expand(y.shape[0], self.n_particles, self.d_particles)
                            temp = self._reshape_backward(temp).squeeze()
                            loss = loss_f1(self._reshape_backward(out).squeeze(), temp)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            stage_loss.append(loss.item()*len(y))

                #print(net_ensemble.boost_rate)
                # store model
                elapsed_tr = time.time()-t0
                sl = 0
                if stage_loss != []:
                    sl = np.sqrt(np.sum(stage_loss)/N)

                print(f'Stage - {ith}, training time: {elapsed_tr: .1f} sec, model MSE loss: {sml: .5f}, Ensemble Net MSE Loss: {sl: .5f}')

                # self.bases.to_file(self.opt.out_f)
                # self.bases = DynamicNet.from_file(self.opt.out_f, lambda stage: MLP_2HL.get_model(stage, self.opt))

                if self.opt.cuda:
                    self.bases.to_cuda()
                self.bases.to_eval() # Set the models in ensemble net to eval mode

                # Train
                tr_rmse = self.root_mse(self.bases, train_loader)
                if self.opt.cv:
                    val_rmse = self.root_mse(self.bases, val_loader)
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_stage = ith

                te_rmse = self.root_mse(self.bases, test_loader)

                print(f'Stage: {ith}  RMSE@Tr: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Te: {te_rmse:.5f}')

                loss_models[ith, 0], loss_models[ith, 1] = tr_rmse, te_rmse

                # P = self.predict(X)
                # G = self.gradient(P, Y)
                # L.fit(X, self._reshape_backward(G))
            else:
                None
                # subsample_idx = self.rng.permutation(X.shape[0])[:int(self.subsample * X.shape[0])]
                # P = self.predict(X[subsample_idx])
                # G = self.gradient(P, Y[subsample_idx])
                # L.fit(X[subsample_idx], self._reshape_backward(G))
            
            # self.bases.append(L)
            # self.rates.append(1.0)
            
    
    def gradient(self, P_: np.ndarray, Y_: np.ndarray) -> np.ndarray:
        if isinstance(P_, np.ndarray):
            P = torch.from_numpy(P_)
        else:
            P = P_
        if isinstance(Y_, np.ndarray):
            Y = torch.from_numpy(Y_).unsqueeze(1).expand(Y_.shape[0], self.n_particles, Y_.shape[1])
        else:
            Y = Y_.unsqueeze(1).expand(Y_.shape[0], self.n_particles, Y_.shape[1])

        gradp = self.grad_logp_vmap(P, Y)
        hessp = self.hess_logp_vmap(P, Y)

        diffs = P.unsqueeze(2) - P.unsqueeze(1)
        kernm = torch.exp(- torch.sum(diffs**2, dim=-1) / self.bandwidth )
        gradk = - (2.0 / self.bandwidth) * diffs * kernm.unsqueeze(3)
        
        phi = torch.mean( kernm.unsqueeze(3) * gradp.unsqueeze(2) + gradk , dim=1 )
        psi = torch.mean( - 1.0 * ( kernm.unsqueeze(3)**2 ) * hessp.unsqueeze(2) + gradk**2 , dim=1 )
        return ( phi / psi ).detach().numpy()
        
    
    def compute_init_base(self, Y_: np.ndarray) -> np.ndarray:
        P0 = torch.from_numpy(self.init_locs)
        Y = torch.from_numpy(Y_).unsqueeze(1).expand(Y_.shape[0], self.n_particles, Y_.shape[1])
        
        for _ in range(self.init_iter):
            gradp = self.grad_logp_vmap(P0.expand(Y.shape[0], P0.shape[0], P0.shape[1]), Y).mean(dim=0)
            hessp = self.hess_logp_vmap(P0.expand(Y.shape[0], P0.shape[0], P0.shape[1]), Y).mean(dim=0)

            diffs = P0.unsqueeze(1) - P0.unsqueeze(0)
            kernm = torch.exp( - torch.sum(diffs**2, dim=-1) / self.bandwidth )
            gradk = - (2.0 / self.bandwidth) * diffs * kernm.unsqueeze(2)
            
            phi = torch.mean( kernm.unsqueeze(2) * gradp.unsqueeze(1) + gradk , dim=0 )
            psi = torch.mean( - 1.0 * ( kernm.unsqueeze(2)**2 ) * hessp.unsqueeze(1) + gradk**2 , dim=0)
            
            P0 += self.init_lr * (phi / psi)
        
        return P0.numpy()
                
    
    def _reshape_forward(self, P: np.ndarray) -> np.ndarray:
        return P.reshape((P.shape[0], self.n_particles, self.d_particles))
    
    
    def _reshape_backward(self, P: np.ndarray) -> np.ndarray:
        return P.reshape((P.shape[0], self.n_particles * self.d_particles))


