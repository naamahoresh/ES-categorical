import numpy as np
import pickle
import sys
import os
import RBFN as RBFN
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
# import logging
# logging.basicConfig(filename='/home/naamah/Documents/CatES/result_All/GECCO/svm_25/logger/logger.log', filemode='a',level=logging.INFO)
import smac_KNN, smac_svm, smac_RF
import time
#import pyKriging
#from pyKriging.krige import kriging

# from pyKriging_master import pyKriging
# from pyKriging_master.pyKriging.krige import kriging



def surrogateRF(Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=False,problem='LABS',index=1):
    Xnew=Xarchive.T
    X_pred=X.T
    SMAC = False

    if SMAC:
        with open("/home/naamah/Documents/CatES/result_All/smac/RF/X1.p", "wb") as fp:
            pickle.dump(Xnew, fp)
        with open("/home/naamah/Documents/CatES/result_All/smac/RF/F1.p", "wb") as fp:
            pickle.dump(Farchive, fp)

        anf=smac_RF.main_loop(problem)

        print("SMAC {}".format(anf))
        sys.exit("Error message")

    clf = RandomForestRegressor(criterion="mse",n_estimators=49, min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0001060554, max_leaf_nodes=1000,
                                    min_impurity_decrease=0.0,min_impurity_split=None, warm_start=False, max_depth=None,
                                    max_features="auto",random_state=7
                                    ) #RF_9111

    # if problem=="LABS":
    #
    #     clf = RandomForestRegressor(criterion="mse",n_estimators=49, min_samples_leaf=1, min_samples_split=2,
    #                                 min_weight_fraction_leaf=0.0001060554, max_leaf_nodes=1000,
    #                                 min_impurity_decrease=0.0,min_impurity_split=None, warm_start=False, max_depth=None,
    #                                 max_features="auto",random_state=7
    #                                 ) #RF_9111
    #
    #
    # elif problem=="NKL":
    #     clf = RandomForestRegressor(criterion="mse",n_estimators=43, min_samples_leaf=1, min_samples_split=4,
    #                                 min_weight_fraction_leaf=0.0005238657425634613, max_leaf_nodes=673, min_impurity_decrease=0.0,
    #                                 warm_start=True, max_depth=None, max_features="auto", random_state=8
    #                                 ) #RF_1_sig
    #
    #
    #
    # else: #problem=="QAP"
    #     clf = RandomForestRegressor(criterion="mse",n_estimators=38, min_samples_leaf=1, min_samples_split=2,
    #                                 min_weight_fraction_leaf=0.0002313685, max_leaf_nodes=551, min_impurity_decrease=2.86E-08,
    #                                 warm_start=False, max_depth=None, max_features="auto", random_state=None
    #                                 )#RF_9333


    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(clf.get_params(), "smac"))
        file.close()

    clf.fit(Xnew, Farchive)
    F_pred = clf.predict(X_pred)

    return F_pred





def surrogateKNN(Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=False,problem='LABS'):
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


    neigh = KNeighborsRegressor(n_neighbors=9,  algorithm="ball_tree", p=1, weights="distance",leaf_size=60) # KNN_LABS_444

    # if problem=="LABS":
    #     neigh = KNeighborsRegressor(n_neighbors=9,  algorithm="ball_tree", p=1, weights="distance",leaf_size=60) # KNN_LABS_444
    #
    #
    #
    # elif problem=="NKL":
    #     neigh = KNeighborsRegressor(n_neighbors=10, algorithm="brute", p=1, weights="distance",leaf_size=76) # KNN_NKL_4442
    #
    #
    #
    # else: #problem=="QAP"
    #     neigh = KNeighborsRegressor(n_neighbors=8,  algorithm="auto", p=1, weights="distance",leaf_size=19) # KNN_QAP_4446


    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(neigh.get_params(), "smac"))
        file.close()


    neigh.fit(Xnew, Farchive)
    F_pred = neigh.predict(X_pred)

    return F_pred





def surrogateSVM(Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=True,problem='LABS'):
    Xnew=Xarchive.T
    X_pred=X.T
    SMAC = False

    if SMAC:
        # smac - no conda
        with open("/home/naamah/Documents/CatES/result_All/smac/svm/X1.p", "wb") as fp:
            pickle.dump(Xnew, fp)
        with open("/home/naamah/Documents/CatES/result_All/smac/svm/F1.p", "wb") as fp:
            pickle.dump(Farchive, fp)

        ans=smac_svm.main_loop(problem)

        print("SMAC {}".format(ans))
        sys.exit("Error message")


        ans_clf=ans.get_dictionary()
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

    # time1=time.time()
    # clf = svm.SVR(kernel='rbf', C=307)
    if problem == "MIS":
        # clf = svm.SVR(kernel='rbf', C=307, epsilon=0.0023, gamma=0.1223 ,shrinking=False,degree=3) #IRACE_SURROGATE SVM_NKL_3331
        clf = svm.SVR(kernel='rbf', C=307, epsilon=0.5, gamma=0.1223 ,shrinking=False,degree=3) #IRACE_SURROGATE SVM_NKL_3331

    else:
        # clf = svm.SVR(kernel='rbf', C=237.689013261001, epsilon=0.00125697429061751, gamma=0.0445112387400382,shrinking=False)  # BYSMAC_9111
        clf = svm.SVR(kernel='rbf', C=237.689013261001, epsilon=0.5, gamma=0.0445112387400382,shrinking=False)  # BYSMAC_9111

    # if problem=="LABS":
    #     # clf = svm.SVR(kernel='rbf', C=307, epsilon=0.0023, gamma=0.1223 ,shrinking=False,degree=3) #R
    #     clf = svm.SVR(kernel='rbf', C=237.689013261001, epsilon=0.00125697429061751, gamma=0.0445112387400382 ,shrinking=False) #BYSMAC_9111
    #
    #
    # elif problem=="NKL":
    #     clf = svm.SVR(kernel='poly', C=999.7180904447123, epsilon=0.0014989615126237686, gamma=1,shrinking=True) #BYSMAC_9111
    #
    # else: #"QAP"
    #     clf = svm.SVR(kernel='rbf', C=307, epsilon=0.0023, gamma=0.1223 ,shrinking=False,degree=3) #IRACE_SURROGATE SVM_NKL_3331


    # time2=time.time()

    if not os.path.exists(file_loc_general + "/surrogate_configuration"):
        with open(file_loc_general + "/surrogate_configuration", 'a') as file:
            file.write("clf:\n{}\n\nTuning Algorithem: {} ".format(clf.get_params(), "smac"))
        file.close()
    clf.fit(Xnew, Farchive)
    # time3=time.time()
    F_pred = clf.predict(X_pred)
    # time4=time.time()
    # logging.warning("\nModel creation: {}  Training: {}  Prediction:  {} \n\n".format(time2-time1, time3-time2, time4-time3))
    return F_pred





def surrogateLM(Xarchive, Farchive, X, toUpdate, test=False, problem='LABS'):
    Fnew = np.asarray(Farchive).reshape((len(Farchive), 1))
    Xnew = Xarchive.T
    X_pred = X.T

    reg = linear_model.LinearRegression()
    reg.fit(Xnew, Fnew)

    F_pred = reg.predict(X_pred)
    F_pred = np.squeeze(F_pred, axis=1)

    return F_pred



def surrogateRBFN (Xarchive, Farchive, X,file_loc,file_loc_general,toUpdate, first_iter=False,problem='LABS',isKmeans=True):
    # http://www.rueckstiess.net/research/snippets/show/72d2363e
    Xnew=Xarchive.T
    X_pred=X.T
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



def surrogateKriging (Xarchive, Farchive, X, file_loc,file_loc_general,toUpdate, first_iter=False,problem='LABS'):
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





#
# X= np.asarray([[1,0,1,0,1,0],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[13,14,15]], dtype=float)
# y= np.asarray([6,15,24,36,42], dtype=float)
# x_test=np.asarray([[1,2,3],[13,14,16],[16,11,8],[16,12,8],[2,3,4],[13,14,15]],dtype=float)
# X= np.asarray([[1,0,1,0,1,0],[1,1,0,0,1,1],[0,0,1,1,1,1],[0,1,1,0,0,0],[0,0,1,0,0,0],[1,1,1,1,1,1],[1,1,0,0,1,1]
#                ,[1,1,1,1,0,1],[0,1,1,1,1,1],[0,0,0,1,0,0]], dtype=float)
# y= np.asarray([3,4,4,2,1,6,4,5,5,1], dtype=float)
# x_test=np.asarray([[1,1,1,0,0,0],[1,1,1,1,1,1],[0,0,0,0,0,0],[0,1,0,1,0,1],[1,0,1,0,1,0]],dtype=float)
# X=X.T
# x_test=x_test.T
# #
# X=np.loadtxt('/home/naamah/Documents/CatES/data/X')
# y=np.loadtxt('/home/naamah/Documents/CatES/data/F')
# x_test=np.loadtxt('/home/naamah/Documents/CatES/data/X_test')
# f_test=np.loadtxt('/home/naamah/Documents/CatES/data/F_test')
# X=X[0:12,:]
# x_test=x_test[0:12,:]
# # x_test=X
# # f_test=y
#
# ans=surrogateKriging(X,y,x_test,file_loc="/home/naamah/Documents/CatES/result_All/result_test/exp_num_1", toUpdate=True,first_iter=True)
# ans2=np.linalg.norm(f_test-ans)
#
# print("the ans for the surrogate is:\n {}\n".format(ans))
# print("the reall fittnes is: \n{}\n".format(f_test))
# print("the diff between the fittnes and the surrogate is:\n {}\n".format(f_test-ans))
# print("the euclidean diff between the fittnes and the surrogate is: {}\n".format(ans2))


