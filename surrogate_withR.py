import numpy as np
#import pyKriging
#from pyKriging.krige import kriging
# from pyKriging_master import pyKriging
# from pyKriging_master.pyKriging.krige import kriging

import pickle
import sys
import os
import RBFN as RBFN
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.cluster import KMeans

#import smac_KNN, smac_svm, smac_RF
from sklearn.model_selection import GridSearchCV
#from hpsklearn import HyperoptEstimator, svr



def surrogateLM(Xarchive, Farchive, X,toUpdate,test=False,problem='Pump'):
    Fnew=np.asarray(Farchive).reshape((len(Farchive),1))
    Xnew=Xarchive.T
    X_pred=X.T

    reg = linear_model.LinearRegression()
    reg.fit(Xnew,Fnew)

    F_pred = reg.predict(X_pred)
    F_pred=np.squeeze(F_pred, axis=1)

    return F_pred




def surrogateRF(Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=False,problem='Pump', RF_parm=None):
    Xnew=Xarchive.T
    X_pred=X.T
    SMAC = False

    if SMAC:
        with open("/home/naamah/Documents/CatES/result_All/X1.p", "wb") as fp:
            pickle.dump(Xnew, fp)
        with open("/home/naamah/Documents/CatES/result_All/F1.p", "wb") as fp:
            pickle.dump(Farchive, fp)

        anf=smac_RF.main_loop(problem)

        print("SMAC {}".format(anf))
        sys.exit("Error message")

    if (RF_parm == None):
        if problem=="Pump":
                clf = RandomForestRegressor(criterion="mse", max_depth=51646, max_features="auto",
                                        max_leaf_nodes=26, min_impurity_decrease=0.0349,
                                        min_samples_leaf=16, min_samples_split=5,
                                        min_weight_fraction_leaf=0.4662, n_estimators=17,
                                        warm_start=False)

        elif problem=="NKL":
                clf = RandomForestRegressor(criterion="mse", max_depth=None,
                                        max_leaf_nodes=673, min_impurity_decrease=0.0,
                                        min_samples_leaf=1, min_samples_split=4,
                                        min_weight_fraction_leaf=0.0005238657425634613, n_estimators=43, random_state=8,
                                        warm_start=True)

        elif problem == "QAP":
                clf = RandomForestRegressor(criterion="mse", max_depth=28444, max_features="sqrt",
                                        max_leaf_nodes=30, min_impurity_decrease=0.3967,
                                        min_samples_leaf=15, min_samples_split=18,
                                        min_weight_fraction_leaf=0.2247, n_estimators=23,
                                        warm_start=True)

    else:
        clf = RandomForestRegressor(criterion=RF_parm.get("criterion"),
                                    max_depth=RF_parm.get("RF_parm"),
                                    max_features=RF_parm.get("max_features"),
                                    max_leaf_nodes=RF_parm.get("max_leaf_nodes"),
                                    min_impurity_decrease=RF_parm.get("min_impurity_decrease"),
                                    min_samples_leaf=RF_parm.get("min_samples_leaf"),
                                    min_samples_split=RF_parm.get("min_samples_split"),
                                    min_weight_fraction_leaf=RF_parm.get("min_weight_fraction_leaf"),
                                    n_estimators=RF_parm.get("n_estimators"),
                                    warm_start=RF_parm.get("warm_start"))

    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(clf.get_params(), "irace"))
        file.close()

    clf.fit(Xnew, Farchive)
    F_pred = clf.predict(X_pred)

    return F_pred



def surrogateKNN(Xarchive, Farchive, X, file_loc, file_loc_general, toUpdate, first_iter=False,problem='Pump', knn_parm=None):
    Xnew=Xarchive.T
    X_pred=X.T
    SMAC = False
    if SMAC:
        with open("/home/naamah/Documents/CatES/result_All/X1.p", "wb") as fp:
            pickle.dump(Xnew, fp)
        with open("/home/naamah/Documents/CatES/result_All/F1.p", "wb") as fp:
            pickle.dump(Farchive, fp)

        anf=smac_KNN.main_loop(problem)

        print("SMAC {}".format(anf))
        sys.exit("Error message")

    if (knn_parm == None):

        if problem=="Pump":
            #neigh = KNeighborsRegressor(n_neighbors=10,  algorithm="ball_tree", p=1, weights="distance",leaf_size=1)
            neigh = KNeighborsRegressor(n_neighbors=10,  algorithm="ball_tree", p=3, weights="distance",leaf_size=10) # R

        elif problem=="NKL":
            neigh = KNeighborsRegressor(n_neighbors=9, algorithm="auto", p=1, weights="distance")

        else: #problem=="QAP"
            # neigh = KNeighborsRegressor(n_neighbors=10,  algorithm="ball_tree", p=1, weights="distance",leaf_size=1)
            neigh = KNeighborsRegressor(n_neighbors=10,  algorithm="auto", p=3, weights="uniform",leaf_size=98) # R

    else:
        neigh = KNeighborsRegressor(n_neighbors=knn_parm.get("n_neighbors"), algorithm=knn_parm.get("algorithm"), p=knn_parm.get("p"), weights=knn_parm.get("weights"), leaf_size=knn_parm.get("leaf_size"))  # R

    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(neigh.get_params(), "irace"))
        file.close()


    neigh.fit(Xnew, Farchive)
    F_pred = neigh.predict(X_pred)

    return F_pred



def surrogateSVM(Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=True,problem='Pump',svm_parm=None):
    Xnew=Xarchive.T
    X_pred=X.T
    SMAC = False

    if SMAC:
        # smac - no conda
        with open("/home/naamah/Documents/CatES/result_All/X1.p", "wb") as fp:
            pickle.dump(Xnew, fp)
        with open("/home/naamah/Documents/CatES/result_All/F1.p", "wb") as fp:
            pickle.dump(Farchive, fp)

        anf=smac_svm.main_loop(problem)

        print("SMAC {}".format(anf))
        sys.exit("Error message")


        ans_clf=anf.get_dictionary()
        C=ans_clf['C']
        epsilon=ans_clf['epsilon']
        kernel=ans_clf['kernel']
        shrinking =True
        if ans_clf['shrinking']=='false':
            shrinking=False
        if ans_clf['gamma']=='value':
            gamma_value = ans_clf['gamma']
        else:
            gamma_value='auto'

    if (svm_parm == None):

        if problem=="Pump":
            #clf = svm.SVR(kernel='rbf', C=237.6890132610009, epsilon=0.0012569742906175072, gamma=0.04451123874003816,shrinking=False) #not sig
            # clf = svm.SVR(kernel='rbf', C=1000, epsilon=0.0012569742906175072, gamma=0.04451123874003816,shrinking=True) #not sig
            clf = svm.SVR(kernel='rbf', C=731, epsilon=0.0010, gamma=0.04451123874003816,shrinking=True) #R


        elif problem=="NKL":
            clf = svm.SVR(kernel='rbf', C=555.7115403997742, epsilon=0.0022692946028747894, gamma=0.8404,degree=1,shrinking=False) #for NKL

        else: #"QAP"
            #clf = svm.SVR(kernel='rbf', C=1000, epsilon=0.03542929048265333, gamma=0.04877700585190937,shrinking=True) #for QAP - sig
            # clf = svm.SVR(kernel='rbf', C=1000, gamma=0.01)
            clf = svm.SVR(kernel='sigmoid', C=785, epsilon=0.0270, gamma=0.0143,shrinking=True, degree=2, coef0=0) #R

    else:
        clf = svm.SVR(kernel=svm_parm.get("kernel"), C=svm_parm.get("C"), epsilon=svm_parm.get("epsilon"), gamma=svm_parm.get("gamma"), degree=svm_parm.get("degree"),
                      shrinking=svm_parm.get("shrinking"),coef0=svm_parm.get("coef0"))

    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(clf.get_params(), "irace"))
        file.close()
    clf.fit(Xnew, Farchive)
    F_pred = clf.predict(X_pred)

    return F_pred



def surrogateRBFN (Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=False,problem='Pump'):
    #1:https://github.com/cybercase/pyradbas/blob/master/pyradbas/train_exact.py
    #2: http://www.rueckstiess.net/research/snippets/show/72d2363e
    Xnew=Xarchive.T
    X_pred=X.T
    isKmeans=True
    Xnew_unique, idx_unique = np.unique(Xnew, axis=0, return_index=True)
    F_unique =Farchive[idx_unique]
    numCenter= int(len(F_unique)*0.75)

    if (toUpdate):
        Rbfn=RBFN.RBF(len(Xnew[0]),numCenter,len(Xnew[0]))
        Rbfn.train(Xnew_unique,F_unique,isKmeans)
        with open(file_loc+"/Rbfn.p", "wb") as fp:
            pickle.dump(Rbfn, fp)
    else:
        with open(file_loc+"/Rbfn.p", "rb") as fp:
            Rbfn = pickle.load(fp)


    F_pred=Rbfn.test(X_pred)
    return F_pred




def surrogateKriging (Xarchive, Farchive, X, file_loc,file_loc_general,toUpdate, first_iter=False,problem='Pump'):
    X_pred=X.T
    F_pred=[]

    # update or build the model
    if (toUpdate):
        Xnew = Xarchive.T
        Xnew_unique, idx_unique = np.unique(Xnew, axis=0, return_index=True)
        F_unique = Farchive[idx_unique]

        if first_iter: #build the model

            testfun = pyKriging.testfunctions().stybtang

            # Now that we have our initial data, we can create an instance of a Kriging model
            k = kriging(Xnew_unique, F_unique, testfunction=testfun, name='simple')
            k.train(optimizer='ga')

        else: #update the model
            with open(file_loc+"kriging.p", "rb") as fp:
                k = pickle.load(fp)
            for point,y_point in zip(Xnew_unique,F_unique):
                k.addPoint(point,y_point)

        #save the new/updates model
        with open(file_loc+"kriging.p", "wb") as fp:
            pickle.dump(k, fp)

    # no new points to update the model
    # only predicts new points.
    else:
        with open(file_loc+"kriging.p","rb") as fp:
            k = pickle.load(fp)

    for x in X_pred:
        F_pred.append(k.predict(x))

    F_pred=np.asarray(F_pred)
    return F_pred


