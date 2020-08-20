# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Feature extractors
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.kernel_approximation import RBFSampler


# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, make_scorer, precision_recall_curve

# helper functions
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
import math
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

# preprocessing steps
# TODO: these should probably be in their own file
import sys
sys.path.insert(1, '../common')
from nlp_utils import process_text
prep_dict = {'inc_not': process_text}
"""
# add here the kind of models we want to run
model_dict = {'Dummy' : DummyClassifier(random_state=3),
              'Stochastic Gradient Descent' : SGDClassifier(random_state=3, loss='log'),
              'Random Forest': RandomForestClassifier(random_state=3),
              'Decsision Tree': DecisionTreeClassifier(random_state=3),
              'AdaBoost': AdaBoostClassifier(random_state=3),
              'K Nearest Neighbor': KNeighborsClassifier()}
"""
# Adaboost parameters
dct = DecisionTreeClassifier(random_state = 11,max_depth = None)
ada_param_grid = {
    'n_estimators' : [50, 100, 300],
    'learning_rate': [0.01, 0.1, 1],
    'base_estimator__max_depth': [1, 3, 5, 15, 25],
    'base_estimator__max_features': [3, 5, 10, 20],
    'base_estimator__class_weight' : ['balanced']
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score),
    'brier_score': make_scorer(brier_score_loss)
}

# SGD paramters
sgd_param_grid = {
    'sgdclassifier__loss': ['modified_huber', 'log'],
    'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
    'sgdclassifier__alpha': [0.01, 0.1, 1],
    'sgdclassifier__max_iter': [500, 1000, 2000],
    'sgdclassifier__class_weight': ['balanced'],
    'sgdclassifier__random_state': [3]
}
# SGD regressor paramters
sgdr_param_grid = {
    'sgdregressor__loss': ['squared_loss', 'huber'],
    'sgdregressor__penalty': ['l2', 'l1', 'elasticnet'],
    'sgdregressor__alpha': [0.01, 0.1, 1],
    'sgdregressor__max_iter': [500, 1000, 2000],
    'sgdregressor__random_state': [3]
}

svc_param_grid = {
    'C': [0.5, 1, 2],
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree' : [2,3,4,5],
    'class_weight': ['balanced'],
    'random_state' : [3]
}
model_dict = {
              'AdaBoost': [AdaBoostClassifier(base_estimator = dct, random_state=3), ada_param_grid, scorers],
              'SGD': [make_pipeline(StandardScaler(with_mean=False), SGDClassifier()), sgd_param_grid, scorers],
              #'SGDK': [make_pipeline(RBFSampler(gamma=1, random_state=1), SGDClassifier()), sgd_param_grid, scorers]
              #'SGDR': [make_pipeline(StandardScaler(with_mean=False), SGDRegressor()), sgdr_param_grid, scorers]
              'SVM': [svm.SVC(probability=True), svc_param_grid, scorers]
              }

              
# add the feature extractors here
feature_dict = {
                'TFIDF-1':  TfidfVectorizer(max_features=500),
                'TFIDF-2': TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=3, max_df=0.98),
                'TFIDF-3': TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=3, max_df=0.98)
                }



# results
def get_results(y_true, y_pred, y_prob):
    ac_score = accuracy_score(y_true, y_pred)
    p_score = precision_score(y_true, y_pred, average='macro')
    r_score = recall_score(y_true, y_pred, average='macro')
    f_1_score = f1_score(y_true,y_pred, average='macro')
    brier_score = brier_score_loss(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    results = { 'accuracy': round(ac_score,4),
                'precision': round(p_score,4),
                'recall': round(r_score,4),
                'f1': round(f_1_score,4),
                'brier': round(brier_score,4),
                'auc': round(auc_score,4)
                }
    return results

# pipeline
def pipeline(preprocessor, vectorizer, vec_name, model_name, model_params , X, y, tt_split=0.2, grid_search = False, preload_data = False ):
    """
    preprocessor : preprocessor for raw text input
    vectorizer: what kind of features we want
    model: model to run and report results on
    X: text input list
    y: target input list
    tt_split:  test / train split

    Returns:
        - F1 score
        - Brier Score
        - accuracy score

    """

    if not preload_data:
        # preprocess the input
        print("Preprocessing data")
        X_prep = [str(preprocessor(x)) for x in X]

        pickle.dump(preprocessor, open('preprocessor.pkl','wb'))

        # vectorize the input 
        print("Vecotirizng data")
        vectorizer.fit(X_prep)
        X_vec = vectorizer.transform(X_prep)
        print("vectorized X shape is: ", X_vec.shape)
        pickle.dump(vectorizer, open(vec_name+'vectorizer.pkl','wb'))

        # split input into test, train
        print("Splitting data into training, and testing data")
        print("X_shape:", X_vec.shape[0], " y_shape:", len(y))
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, stratify =y, random_state=3)

        pickle.dump(X_train, open('X_train.pkl', 'wb'))
        pickle.dump(X_test, open('X_test.pkl', 'wb'))
        pickle.dump(X_train, open('y_train.pkl', 'wb'))
        pickle.dump(X_test, open('y_test.pkl', 'wb'))
    else:
        X_train = pickle.load(open('X_train.pkl', 'rb'))
        X_test = pickle.load(open('X_test.pkl', 'rb'))
        y_train = pickle.load(open('y_train.pkl', 'rb'))
        y_test = pickle.load(open('y_test.pkl', 'rb'))

    model = model_params[0]
    param_grid = model_params[1]
    scorers =  model_params[2]

    if grid_search:
        #if not (os.path.exists((model_name+'.pkl'))):
        print("Performing grid search")
        # optinally search the grid for the best model?
        #grid_search_ABC = GridSearchCV(model, param_grid=param_grid, scoring = 'roc_auc')
        
        # TODO: there's duplication of code here.. need to clean it up
        skf = StratifiedKFold(n_splits=5)

        """
        precision_gs = GridSearchCV(model, param_grid=param_grid, scoring=scorers, refit='precision_score',
                            cv=skf, return_train_score=True, n_jobs=-1)
        
        
        precision_gs.fit(X_train, y_train)

        # make the predictions
        y_pred = precision_gs.predict(X_test)

        print('Best params for precision_score:')
        print(precision_gs.best_params_)
        #print("Results of precision GS :", precision_gs.cv_results_)

        # confusion matrix on the test data.
        print('\nConfusion matrix of {} optimized for precision_score on the test data:'.format(model_name))
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                    columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        """

        f1_gs = GridSearchCV(model, param_grid=param_grid, scoring=scorers, refit='f1_score',
                            cv=skf, return_train_score=True, n_jobs=-1)
        f1_gs.fit(X_train, y_train)

        # make the predictions
        y_pred = f1_gs.predict(X_test)

        print('Best params for brier_score:')
        print(f1_gs.best_params_)
        #print("Results of brier GS :", f1_gs.cv_results_)


        # confusion matrix on the test data.
        print('\nConfusion matrix of {} optimized for brier_score on the test data:'.format(model_name))
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                    columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

        model = f1_gs.best_estimator_
        pickle.dump(model, open(model_name+'_'+vec_name+'.pkl','wb'))
        """
        else:
            print("Loading pre-searched model")
            model = pickle.load(open(model_name+'.pkl','rb'))
        """

    else:
        # train the model
        print("Training the model..")
        model.fit(X_train, y_train)

    # predict the target/label
    print(" making model predictions")
    y_pred = model.predict(X_test)

    #print("Y_test:", y_test)
    #print("Y_pred:", y_pred)

    # get the probabilities of prediction
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except AttributeError:
        y_prob = model.decision_function(X_test)

    #p, r, thresholds = precision_recall_curve(y_test, y_prob)
    #plot_precision_recall_vs_threshold(p,r,thresholds)
    #print(y_prob)
    # get the results after making predictions
    results = get_results(y_test, y_pred, y_prob)
    
    print(results)
    return results


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.show()

# run all the pipelines, given data
def run_all_pipelines(X,y, grid_search=False):
    results = []
    for prep_k, prep_v in prep_dict.items():
        for model_k,model_v in model_dict.items():
            for feat_k, feat_v in feature_dict.items():
                c_result = {}
                c_result['model'] = model_k
                c_result['prep'] = prep_k                
                c_result['vectorizer'] = feat_k                
                print("Running pipeline for:", prep_k, model_k, feat_k)
                p_result = pipeline(prep_v, feat_v, feat_k, model_k, model_v, X, y, 0.2, grid_search)
                c_result.update(p_result)

                results.append(c_result)
    return results
