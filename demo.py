import numpy as np, scipy, torch
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from src.swgboost import SWGBoost

X = np.linspace(-3.5, 3.5, 200).reshape(-1,1)                   # input data
Y = np.sin(X)
D = scipy.stats.norm(loc=Y.flatten(), scale=0.5)                # target distributions N( p | m=y_i, s=0.5 ) conditional on each y_i in Y = sin(X)

grad_logp = lambda p, y: - (p - y) / 0.5**2                     # define the log gradient of the target distribution N( p | m=y, s=0.5 ) at a location p conditional on a value y
hess_logp = lambda p, y: - torch.ones(1) / 0.5**2               # define the log hessian diagonal of the target distribution N( p | m=y, s=0.5 ) at a location p conditional on a value y
reg = SWGBoost(grad_logp, hess_logp, DecisionTreeRegressor,     # use DecisionTreeRegressor as each base learner
               learner_param = {'max_depth': 3, 'random_state': 1},        # pass hyperparameters to DecisionTreeRegressor
               learning_rate = 0.1,                                        # set the learning rate
               n_estimators = 100,                                         # set the number of base learners to be used
               n_particles = 10,                                           # set the number of output particles of SWGBoost
               d_particles = 1,                                            # inform the dimension of each output particle of SWGBoost
               init_iter = 0,                                              # no optimisation for the initial state of SWGBoost
               init_locs = np.linspace(-10, 10, 10).reshape(-1, 1))        # set the initial state of SWGBoost
reg.fit(X, Y)                                                   # fit SWGBoost
P = reg.predict(X)                                              # predict by SWGBoost

fig, ax = plt.subplots(figsize=(8,4))                           # plot the output
ax.fill_between(X.flatten(), D.ppf(0.025), D.ppf(0.975), color='b', alpha=0.05)
[ sns.lineplot(x=X.flatten(), y=P[:,ith].flatten(), color="red", ax=ax) for ith in range(reg.n_particles) ]
fig.tight_layout()
fig.show()
