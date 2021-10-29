from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('RF',RandomForestClassifier()))
results = []
names = []
scoring = ['precision_macro', 'recall_macro','accuracy','f1_macro']


for name, model in models:
    kfold = KFold(n_splits=10,shuffle=True,random_state=10)
    cv_results = cross_validate(model, X_res, y_res, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s -> %s: %f %s: %f" % (name,"MEAN Accuracy", cv_results['test_accuracy'].mean(),"STD" ,cv_results['test_accuracy'].std())
    print(msg)
    msg = "%s -> %s: %f %s: %f" % (name,"MEAN PRECISION", cv_results['test_precision_macro'].mean(),"STD" ,cv_results['test_precision_macro'].std())
    print(msg)
    msg = "%s -> %s: %f %s: %f" % (name,"MEAN Recall", cv_results['test_recall_macro'].mean(),"STD" ,cv_results['test_recall_macro'].std())
    print(msg)
    msg = "%s -> %s: %f %s: %f" % (name,"MEAN F1-score", cv_results['test_f1_macro'].mean(),"STD" ,cv_results['test_f1_macro'].std())
    print(msg)
    print("_"*50)
