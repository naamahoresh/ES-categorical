
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
import time
import datetime
# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
import surrogate
import pickle

# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()

def svm_from_cfg(cfg):
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"


    clf = svm.SVR(**cfg)
    with open("/home/naamah/Documents/CatES/result_All/smac/svm/X1.p", "rb") as fp:
        X=pickle.load(fp)
    with open("/home/naamah/Documents/CatES/result_All/smac/svm/F1.p", "rb") as fp:
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
    #logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    cs = ConfigurationSpace()

    kernel = CategoricalHyperparameter("kernel", ["rbf", "sigmoid"], default_value="rbf")
    cs.add_hyperparameter(kernel)

    C = UniformFloatHyperparameter("C", 1, 1000.0, default_value=900.0)
    shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="false")
    cs.add_hyperparameters([C, shrinking])

    #degree = UniformIntegerHyperparameter("degree", 1, 3 ,default_value=3)     # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 1.0, default_value=0.0)  # poly, sigmoid
    #cs.add_hyperparameters([degree, coef0])
    cs.add_hyperparameter(coef0)
    #use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0, parent=kernel, values=["sigmoid"])
    #cs.add_conditions([use_degree, use_coef0])
    cs.add_condition(use_coef0)

    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.001, 8, default_value=1)
    cs.add_hyperparameters([gamma, gamma_value])
    cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
    cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "sigmoid"]))

    epsilon = UniformFloatHyperparameter("epsilon", 0.001, 5.0, default_value=0.1)
    cs.add_hyperparameter(epsilon)

    # Scenario object
    max_eval=100000
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_eval,  # maximum function evaluations
                         "cs": cs,                        # configuration space
                         "shared_model": True,
                         "output_dir": "/home/naamah/Documents/CatES/result_All/smac/svm/run_{}_{}_{}".format(max_eval,
                                                                                                           datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'),
                                                                                                              problem),
                         # "input_psmac_dirs": "/home/naamah/Documents/CatES/result_All/smac/svm/{}/run_{}_{}/*".format(problem,max_eval, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_%f')),
                         "deterministic": "False"
                         #"instance_file":"/home/naamah/Documents/CatES/result_All",
                        #"test_instance_file":"/home/naamah/Documents/CatES/result_All"
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
    # We can also validate our results (though this makes a lot more sense with instances)
    #smac.validate(config_mode='inc',repetitions=100,n_jobs=1) # How many cores to use in parallel for optimization


