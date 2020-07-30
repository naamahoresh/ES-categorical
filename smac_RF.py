
import logging
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import time
import datetime

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, LessThanCondition, GreaterThanCondition

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
import pickle
from sklearn.ensemble import RandomForestRegressor


iris = datasets.load_iris()

def svm_from_cfg(cfg):
    cfg = {k : cfg[k] for k in cfg if cfg[k]}

    cfg["warm_start"] = True if cfg["warm_start"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "max_depth" in cfg:
        cfg["max_depth"] = cfg["max_depth_value"] if cfg["max_depth"] == "value" else None
        cfg.pop("max_depth_value", None)

    if "random_state" in cfg:
        cfg["random_state"] = cfg["random_state_value"] if cfg["random_state"] == "value" else None
        cfg.pop("random_state_value", None)

    if "max_features" in cfg:
        if cfg["max_features"] == "int":
            cfg["max_features"] = cfg.get("max_features_int",'auto')
        elif cfg["max_features"] == "float":
            cfg["max_features"] = cfg.get("max_features_float",'auto')
        elif cfg["max_features"] == "None":
            cfg["max_features"] = None
        else:
            cfg["max_features"] = cfg["max_features"]

        cfg.pop("max_features_int", None)
        cfg.pop("max_features_float", None)

    clf = RandomForestRegressor(**cfg)

    with open("/home/naamah/Documents/CatES/result_All/smac/RF/X1.p", "rb") as fp:
        X=pickle.load(fp)
    with open("/home/naamah/Documents/CatES/result_All/smac/RF/F1.p", "rb") as fp:
        Y=pickle.load(fp)
    # X=iris.data
    # Y=iris.target
    #
    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y) ** 2))

    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    scores = cross_val_score(clf, X,Y, cv=5,scoring=rmse_scorer)

    return (-1)*np.mean(scores) # Because cross_validation sign-flips the score

    # scores = cross_val_score(clf, X,Y, cv=5)
    # return 1-np.mean(scores) # Minimize!


def main_loop(problem):
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
    cs = ConfigurationSpace()

    n_estimators = UniformIntegerHyperparameter("n_estimators", 5,50, default_value=10)
    #criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0)
    max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)
    min_impurity_decrease = UniformFloatHyperparameter("min_impurity_decrease", 0.0, 0.5, default_value=0.0)
    warm_start = CategoricalHyperparameter("warm_start", ["true", "false"], default_value="false")

    cs.add_hyperparameters([n_estimators,min_weight_fraction_leaf, min_samples_split, min_samples_leaf, max_leaf_nodes,warm_start,min_impurity_decrease])

    max_features = CategoricalHyperparameter("max_features", ["auto","log2","sqrt", "int","None","float"], default_value="auto")  # only rbf, poly, sigmoid
    max_features_int = UniformIntegerHyperparameter("max_features_int", 2, len(X[0]), default_value=5)
    max_features_float = UniformFloatHyperparameter("max_features_float", 0.0, 0.9, default_value=0.0)
    cs.add_hyperparameters([max_features, max_features_int, max_features_float])
    use_max_features_int = InCondition(child=max_features_int, parent=max_features, values=["int"])
    use_max_features_float= InCondition(child=max_features_float, parent=max_features, values=["float"])
    cs.add_conditions([use_max_features_int,use_max_features_float])

    max_depth = CategoricalHyperparameter("max_depth",["None","value"], default_value="None")
    max_depth_value = UniformIntegerHyperparameter("max_depth_value", 2, 20, default_value=5)
    cs.add_hyperparameters([max_depth, max_depth_value])
    cs.add_condition(InCondition(child=max_depth_value, parent=max_depth, values=["value"]))

    random_state = CategoricalHyperparameter("random_state", ["None", "value"], default_value="None")
    random_state_value = UniformIntegerHyperparameter("random_state_value", 1, 20, default_value=1)
    cs.add_hyperparameters([random_state, random_state_value])
    cs.add_condition(InCondition(child=random_state_value, parent=random_state, values=["value"]))

    with open("/home/naamah/Documents/CatES/result_All/X1.p", "rb") as fp:
        X = pickle.load(fp)

    # Scenario object
    max_eval=100000
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": max_eval,  # maximum function evaluations
                         "cs": cs,                        # configuration space
                         "shared_model": True,
                         "output_dir": "/home/naamah/Documents/CatES/result_All/smac/RF/run_{}_{}_{}".format(max_eval,
                                                                                                           datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S'),
                                                                                                              problem),
                         "input_psmac_dirs": "/home/naamah/Documents/CatES/result_All/smac/psmac",
                         "deterministic": "False"
                         })

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