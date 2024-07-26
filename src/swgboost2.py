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
from torch.func import grad, jacrev, vmap
from typing import List, Optional, Callable

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from sklearn.svm import NuSVR, SVC, SVR, LinearSVR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoCV, RidgeCV, LarsCV, OrthogonalMatchingPursuitCV, LassoLarsCV, ElasticNet, ElasticNetCV, SGDRegressor, LassoLars, Lasso, Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from time import perf_counter
from sklearn.metrics import mean_squared_error, r2_score, f1_score
import warnings

warning_flag = 0

#==========================================
# Main Classes
#==========================================
class SWGBoost2(BaseEstimator):
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
            init_iter: float = 50,
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

        # self._models_lst = {DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
        #                     ElasticNetCV, LassoLarsCV, LassoCV,
        #                     NuSVR, SGDRegressor, KernelRidge, MLPRegressor,
        #                     RidgeCV, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, LassoLarsIC}
        # self._models_lst = {DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
        #                     ElasticNet, LassoLars, Lasso, SGDRegressor,
        #                     Ridge, ARDRegression, RANSACRegressor, LassoLarsIC}
        self._models_lst = {DecisionTreeRegressor,}
        self._models = deepcopy(self._models_lst)
        self.custom_loss_metrics = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((X.shape[0], self.n_particles, self.d_particles))
        P += self.base0
        for ith in range(len(self.bases)):
            P += self.learning_rate * self.rates[ith] * self._reshape_forward( self.bases[ith].predict(X) )
        return P
    
    
    def predict_eachitr(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((self.n_estimators, X.shape[0], self.n_particles, self.d_particles))
        P[0] = self.base0 + self.learning_rate * self.rates[0] * self._reshape_forward( self.bases[0].predict(X) )
        for ith in range(1, len(self.bases)):
            P[ith] = P[ith-1] + self.learning_rate * self.rates[ith] * self._reshape_forward( self.bases[ith].predict(X) )
        return P


    def _metrics(self, vt, vp, model, time=None):
        """get loss metrics of a model

        Args:
            vt (iterable): validation true values
            vp (iterable): validation pred values
            model (object): any model with fit and predict method
            time (float, optional): execution time of the model. Defaults to None.

        Returns:
            dict['model', 'time', 'loss']
        """
        if self.custom_loss_metrics:
            return {'model': model, 'time': time, 'loss': self.custom_loss_metrics(vt, vp)}
        return {"model": model, "time": time, "loss": mean_squared_error(vt, vp)}


    def _create_model(self, X, y, model_name, time_it: bool = False):
        """fit a model instance

        Args:
            X (iterable)
            y (iterable)
            model_name (object): any model object with fit and predict methods
            time_it (bool, optional): measure execution time. Defaults to False.

        Returns:
            tuple(model, time=None)
        """
        model = model_name()
        if time_it:
            begin = perf_counter()
            model.fit(X, y)
            end = perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)


    def _get_metrics(self, model_name):
        """a helper fuction, combines self._create_model and self._metrics

        Args:
            model_name (object): any model with fit and predict methods

        Returns:
            self._metrics
        """
        try:
            # Xt, Xv, yt, yv = train_test_split(self._X, self._y, test_size=0.2)
            # results = self._create_model(Xt, yt, model_name, time_it=False)
            with warnings.catch_warnings(record=True) as w:
                Xt, Xv, yt, yv = train_test_split(self._X, self._y, test_size=0.2)
                results = self._create_model(Xt, yt, model_name, time_it=False)
                if len(w) > 0 and warning_flag == 0:
                    print(Xt.shape, len(yt), model_name)
                    print(yv)
                    warning_flag += 1
            model, time = results[0], results[1]
            return self._metrics(yv,
                                 model.predict(Xv), model, time)
        except Exception:
            return None


    def _get_results(self, X, y) -> list:
        """Use multi-threading to return all results

        Args:
            X (iterable)
            y (iterable)

        Returns:
            list[dict['model', 'time', 'loss']]
        """
        results = []
        # self._X = self._minimax.fit_transform(self._robust.fit_transform(
        #         KNNImputer(weights='distance').fit_transform(X)))
        self._X = X
        self._y = y
        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            res = executor.map(self._get_metrics, self._models)
            results = [i for i in res if i]
        return results


    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.base0 = self.compute_init_base(Y)


        for ith in range(self.n_estimators):
            L = self.learner_class(**self.learner_param)
            
            if self.subsample == 1.0:
                P = self.predict(X)
                G = self.gradient(P, Y)

                results = self._get_results(X, self._reshape_backward(G))
                try:
                    min_loss = min(results, key=lambda x: x.get(
                        "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                except Exception:
                    continue
                L = [i['model']
                             for i in results if min_loss >= i['loss']][0]

                # L.fit(X, self._reshape_backward(G))
            else:
                subsample_idx = self.rng.permutation(X.shape[0])[:int(self.subsample * X.shape[0])]
                P = self.predict(X[subsample_idx])
                G = self.gradient(P, Y[subsample_idx])

                results = self._get_results(X, self._reshape_backward(G))
                try:
                    min_loss = min(results, key=lambda x: x.get(
                        "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                except Exception:
                    continue
                L = [i['model']
                     for i in results if min_loss >= i['loss']][0]
                # L.fit(X[subsample_idx], self._reshape_backward(G))
            
            self.bases.append(L)
            self.rates.append(1.0)
            
    
    def gradient(self, P_: np.ndarray, Y_: np.ndarray) -> np.ndarray:
        P = torch.from_numpy(P_)
        Y = torch.from_numpy(Y_).unsqueeze(1).expand(Y_.shape[0], self.n_particles, Y_.shape[1])

        gradp = self.grad_logp_vmap(P, Y)
        hessp = self.hess_logp_vmap(P, Y)

        diffs = P.unsqueeze(2) - P.unsqueeze(1)
        kernm = torch.exp(- torch.sum(diffs**2, dim=-1) / self.bandwidth )
        gradk = - (2.0 / self.bandwidth) * diffs * kernm.unsqueeze(3)
        
        phi = torch.mean( kernm.unsqueeze(3) * gradp.unsqueeze(2) + gradk , dim=1 )
        psi = torch.mean( - 1.0 * ( kernm.unsqueeze(3)**2 ) * hessp.unsqueeze(2) + gradk**2 , dim=1 )
        return ( phi / psi ).numpy()
        
    
    def compute_init_base(self, Y_: np.ndarray) -> None:
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


