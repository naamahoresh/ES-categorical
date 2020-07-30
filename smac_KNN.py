
import logging
import numpy as np
from sklearn import neighbors as neighbor, datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import time
import datetime

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
import surrogate
import pickle


def svm_from_cfg(cfg):
    cfg = {k : cfg[k] for k in cfg if cfg[k]}

    clf = neighbor.KNeighborsRegressor(**cfg, n_jobs =-1)
    with open("/home/naamah/Documents/CatES/result_All/X1.p", "rb") as fp:
        X=pickle.load(fp)
    with open("/home/naamah/Documents/CatES/result_All/F1.p", "rb") as fp:
        Y=pickle.load(fp)

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y) ** 2))
        #root mean square error

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    scores = cross_val_score(clf, X,Y, cv=5,scoring=rmse_scorer)
    return (-1)*np.mean(scores) # Because cross_validation sign-flips the score

    # scores = cross_val_score(clf, X,Y, cv=5)
    # return 1-np.mean(scores) # Minimize!

def main_loop(problem):
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    cs = ConfigurationSpace()

    n_neighbors = UniformIntegerHyperparameter("n_neighbors", 2,10, default_value=5)
    cs.add_hyperparameter(n_neighbors)

    weights = CategoricalHyperparameter("weights", ["uniform","distance"], default_value="uniform")
    algorithm = CategoricalHyperparameter("algorithm", ["ball_tree", "kd_tree","brute","auto"], default_value="auto")
    cs.add_hyperparameters([weights, algorithm])

    leaf_size = UniformIntegerHyperparameter("leaf_size", 1, 100, default_value=50)
    cs.add_hyperparameter(leaf_size)
    use_leaf_size= InCondition(child=leaf_size, parent=algorithm, values=["ball_tree","kd_tree"])
    cs.add_condition(use_leaf_size)

    p = UniformIntegerHyperparameter("p", 1,3, default_value=2)
    cs.add_hyperparameter(p)

    # Scenario object
    max_eval=100000
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_eval,  # maximum function evaluations
                         "cs": cs,                        # configuration space
                         "shared_model": True,
                         "output_dir": "/home/naamah/Documents/CatES/result_All/smac/KNN/run_{}_{}_{}".format(max_eval,
                                                                                                           datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'),
                                                                                                              problem)

                         # "output_dir": "/home/naamah/Documents/CatES/result_All/smac/KNN/{}/run_{}_{}".format(problem,max_eval, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')),
                         # "input_psmac_dirs":"/home/naamah/Documents/CatES/result_All/",
                         # "deterministic": "true"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = svm_from_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC(scenario=scenario,tae_runner=svm_from_cfg)

    incumbent = smac.optimize()

    inc_value = svm_from_cfg(incumbent)
    print("Optimized Value: %.2f" % (inc_value))

    return (incumbent)


# main_loop()