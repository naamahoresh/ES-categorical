import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm



def surrogateLM(Xarchive, Farchive, X,toUpdate,test=False,problem='Pump'):
    Fnew=np.asarray(Farchive).reshape((len(Farchive),1))
    Xnew=Xarchive.T
    X_pred=X.T

    reg = linear_model.LinearRegression()
    reg.fit(Xnew,Fnew)

    F_pred = reg.predict(X_pred)
    F_pred=np.squeeze(F_pred, axis=1)

    return F_pred


def surrogateRF(Xarchive, Farchive, X,problem='Pump'):
    Xnew=Xarchive.T
    X_pred=X.T

    if problem=="Pump":
        clf = RandomForestRegressor(criterion="mse",n_estimators=49, min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0001060554, max_leaf_nodes=1000,
                                    min_impurity_decrease=0.0,min_impurity_split=None, warm_start=False, max_depth=None,
                                    max_features="auto",random_state=7
                                    ) #RF_9111
    elif problem=="NKL":
        clf = RandomForestRegressor(criterion="mse",n_estimators=43, min_samples_leaf=1, min_samples_split=4,
                                    min_weight_fraction_leaf=0.0005238657425634613, max_leaf_nodes=673, min_impurity_decrease=0.0,
                                    warm_start=True, max_depth=None, max_features="auto", random_state=8
                                    ) #RF_1_sig

    else: #problem=="QAP"
        clf = RandomForestRegressor(criterion="mse",n_estimators=38, min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0002313685, max_leaf_nodes=551, min_impurity_decrease=2.86E-08,
                                    warm_start=False, max_depth=None, max_features="auto", random_state=None
                                    )#RF_9333

    clf.fit(Xnew, Farchive)
    F_pred = clf.predict(X_pred)

    return F_pred



def surrogateKNN(Xarchive, Farchive, X,problem='Pump'):
    Xnew=Xarchive.T
    X_pred=X.T

    if problem=="Pump":
        neigh = KNeighborsRegressor(n_neighbors=9,  algorithm="ball_tree", p=1, weights="distance",leaf_size=60) # KNN_Pump_444

    elif problem=="NKL":
        neigh = KNeighborsRegressor(n_neighbors=10, algorithm="brute", p=1, weights="distance",leaf_size=76) # KNN_NKL_4442

    else: #problem=="QAP"
        neigh = KNeighborsRegressor(n_neighbors=8,  algorithm="auto", p=1, weights="distance",leaf_size=19) # KNN_QAP_4446

    neigh.fit(Xnew, Farchive)
    F_pred = neigh.predict(X_pred)

    return F_pred



def surrogateSVM(Xarchive, Farchive, X,problem='Pump'):
    Xnew=Xarchive.T
    X_pred=X.T

    if problem=="Pump":
        # clf = svm.SVR(kernel='rbf', C=307, epsilon=0.0023, gamma=0.1223 ,shrinking=False,degree=3) #R
        clf = svm.SVR(kernel='rbf', C=237.689013261001, epsilon=0.00125697429061751, gamma=0.0445112387400382 ,shrinking=False) #BYSMAC_9111

    elif problem=="NKL":
        clf = svm.SVR(kernel='poly', C=999.7180904447123, epsilon=0.0014989615126237686, gamma=1,shrinking=True) #BYSMAC_9111

    else: #"QAP"
        clf = svm.SVR(kernel='rbf', C=307, epsilon=0.0023, gamma=0.1223 ,shrinking=False,degree=3) #IRACE_SURROGATE SVM_NKL_3331

    clf.fit(Xnew, Farchive)
    F_pred = clf.predict(X_pred)

    return F_pred


