import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from time import time_ns
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def load_dataset(file_name):
    return pd.read_csv(file_name)

def scale_ds(ds, scale=False):
    if not scale:
        return ds
    num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])
    ds_scale = num_pipeline.fit_transform(ds)
    return ds_scale

def plot_data(X, Y, x_label, y_label, title, labels, figure_n=1, sub=''):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    for i in range(len(Y)):
        if type(X[0]) == np.int32:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(X, Y[i], marker='o', label=labels[i])
    ax.legend()
    ax.grid(True)
    plt.savefig(f'figure_{figure_n}{sub}.png', dpi='figure')
    
def plat_subgraph(source_data, figure_n=1):
    fig, ax = plt.subplots(1, len(source_data), figsize=(18, 6))
    #plt.tight_layout()
    sub_c = ['a', 'b', 'c', 'd']
    for i in range(len(source_data)):
        X = source_data[i]['X']
        Y = source_data[i]['Y']
        if type(X[0]) == np.int32:
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].set_title(f"{sub_c[i]}. {source_data[i]['title']}", fontsize=16)
        ax[i].set_xlabel(source_data[i]['axis_lablels'][0], fontsize=14)
        ax[i].set_ylabel(source_data[i]['axis_lablels'][1], fontsize=14)
        for j in range(len(Y)):
            ax[i].plot(X, Y[j], marker='o', label=source_data[i]['labels'][j])
        ax[i].legend()
        ax[i].grid(True)
    plt.savefig(f'figure_{figure_n}.png', dpi='figure')

def plot_hist(data, x_label, y_label, title, file_name='figure_1', lables=None):
    ones = len(data[data==1])
    zeros = len(data[data==0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(lables, fontsize=14)
    ax.bar(lables, [zeros, ones], width=0.2)
    #ax.hist(data)
    plt.savefig(file_name+'.png', dpi='figure')

def prepare_bank_dataset():
    ds = load_dataset("datasets/BankChurners.csv")
    ds1 = ds.drop(ds.columns[-2:],axis=1)
    ds1 = ds1.drop(columns=['CLIENTNUM'])
    cat_ds1 = ds1.select_dtypes(include=['object']).copy()
    Attrition_map = {'Attrition_Flag': {'Existing Customer': 0, 'Attrited Customer': 1}}
    cat_ds1.replace(Attrition_map, inplace=True)
    Gender_map = {'Gender': {'F': 1, 'M': 2}}
    cat_ds1.replace(Gender_map, inplace=True)
    Education_map = {'Education_Level': {'College': 3, 'Doctorate': 6, 'Graduate': 4, 
                                         'High School': 2, 'Post-Graduate': 5, 'Uneducated': 1, 'Unknown': 3}}
    cat_ds1.replace(Education_map, inplace=True)
    card_map = {'Card_Category': {'Blue': 1, 'Gold': 3, 'Platinum': 4, 'Silver': 2}}
    cat_ds1.replace(card_map, inplace=True)
    #marry_map = {'Marital_Status': {'Divorced': 1, 'Married': 2, 'Single': 3, 'Unknown': 4}}
    marry_map = {'Marital_Status': {'Divorced': 2, 'Married': 3, 'Single': 1, 'Unknown': 2}}
    cat_ds1.replace(marry_map, inplace=True)
    income_map = {'Income_Category': {'$120K +': 5, '$40K - $60K': 2, '$60K - $80K': 3,
                                     '$80K - $120K': 4, 'Less than $40K': 1, 'Unknown': 3}}
    cat_ds1.replace(income_map, inplace=True)
    ds1[list(cat_ds1.columns)] = cat_ds1
    
    return ds1

def comp_final_test(org_clf, final_clf, X_train, y_train, X_test, y_test, score_func=recall_score, down_sample=False,
                    down_X=None, down_y=None):
    org_train_scores, org_test_scores, org_train_time, org_test_time, ora_clf_r, c_matrix = \
        final_test(org_clf, X_train, y_train, X_test, y_test, score_func=score_func)
    print("Un-tuned model score for Heart Disease Dataset:")
    if type(ora_clf_r) == DecisionTreeClassifier:
        print(f"Tree Node: {ora_clf_r.tree_.node_count}", end=', ')
    print(f"Train Score: {org_train_scores}, Test Score: {org_test_scores}, \
          Train Time: {org_train_time}, Test Time: {org_test_time}")
    print("Confusion matrix:", c_matrix)
    
    if down_sample:
        final_train_scores, final_test_scores, final_train_time, final_test_time, final_clf_r, _ = \
            final_test(final_clf, down_X, down_y, X_test, y_test, score_func=score_func)
    else:
        final_train_scores, final_test_scores, final_train_time, final_test_time, final_clf_r, _ = \
            final_test(final_clf, X_train, y_train, X_test, y_test, score_func=score_func)
    
    print("Final model score for Heart Disease Dataset:")
    if type(final_clf_r) == DecisionTreeClassifier:
        print(f"Tree Node: {final_clf_r.tree_.node_count}", end=', ')
    print(f"Train Score: {final_train_scores}, Test Score: {final_test_scores}, \
          Train Time: {final_train_time}, Test Time: {final_test_time}\n")

def final_test(clf, X_train, y_train, X_test, y_test, score_func=recall_score):
    time_start = time_ns()
    clf.fit(X_train, y_train)
    train_time = time_ns() - time_start
    time_start = time_ns()
    predict_tran_y = clf.predict(X_train)
    test_time = time_ns() - time_start
    predict_test_y = clf.predict(X_test)
    train_scores = round(score_func(y_train, predict_tran_y), 4)
    test_scores = round(score_func(y_test, predict_test_y), 4)
    c_matrix = classification_report(y_test, predict_test_y)
    return train_scores, test_scores, train_time, test_time, clf, c_matrix
    

def learn_curve_test(clf, X_train, y_train, X_test, y_test, score_func=recall_score, random_state=10):
    sample_size_train_scroes = [] 
    sample_size_test_scroes = []
    size_ratio = [.9, .8, .7, .6, .5, .4, .3, .2, .1, 0]
    for ratio in size_ratio:
        if ratio == 0:
            sub_X_train, sub_y_train = X_train, y_train
        else:
            sub_X_train, _, sub_y_train, _ = \
                train_test_split(X_train, y_train, test_size=ratio, random_state=random_state, shuffle=True, stratify=y_train)
        #clf = make_pipeline(StandardScaler(), SVC(**best_arg))
        clf.fit(sub_X_train, sub_y_train)
        predict_tran_y = clf.predict(sub_X_train)
        predict_test_y = clf.predict(X_test)
        avg_train_scores_roc = score_func(sub_y_train, predict_tran_y)
        avg_test_scores_roc = score_func(y_test, predict_test_y)
        sample_size_train_scroes.append(avg_train_scores_roc)
        sample_size_test_scroes.append(avg_test_scores_roc)
    size_ratio_plot = ((1 - np.array(size_ratio)) * 100).astype(np.int)
    return size_ratio_plot, sample_size_train_scroes, sample_size_test_scroes

def resample_Xy(X, y, up_sample=False, apply=False):
    if not apply:
        return X, y
    
    ind_0 = np.where(y==0)
    ind_1 = np.where(y==1)
    
    if up_sample:
        ds_1 = X[ind_1]
        ds_1_y = y[ind_1]
        ds_2 = X[ind_0]
        ds_2_y = y[ind_0]
    else:
        ds_1 = X[ind_0]
        ds_1_y = y[ind_0]
        ds_2 = X[ind_1]
        ds_2_y = y[ind_1]
        
    new_idx = np.random.choice(len(ds_1), len(ds_2), replace=up_sample)

    X = np.concatenate((ds_2, ds_1[new_idx]))
    y = np.concatenate((ds_2_y, ds_1_y[new_idx]))
    
    return X, y

def cross_validate_func(learner, X_train, y_train, init_args, arg_name, arg_values, scoring, 
                        cv=5, verbose=False, scale_data=False, random_state=100, learner2=None, init_args2=None, arg_test=0):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    train_scores = []
    test_scores = []
    train_time = []
    test_time = []
    estimators = []
    for value in arg_values:
        if arg_test == 0:
            init_args[arg_name] = value
        else:
            init_args2[arg_name] = value
        if scale_data:
            #print("here!")
            clf = make_pipeline(StandardScaler(), learner(**init_args))
        elif learner2 == None:
            clf = learner(**init_args)
        else:
            clf = learner(learner2(**init_args2), **init_args)
        scores = cross_validate(clf, X_train, y_train, cv=kfold, scoring=scoring,
                                return_train_score=True, return_estimator=True)
        if verbose:
            print(value, scores['train_score'].mean(),scores['test_score'].mean())
        train_scores.append(scores['train_score'].mean())
        test_scores.append(scores['test_score'].mean())
        train_time.append(scores['fit_time'].mean())
        test_time.append(scores['score_time'].mean())
        estimators.append(scores['estimator'])

    if verbose:
        idx = np.array(test_scores).argmax()
        print(arg_values[idx], train_scores[idx], test_scores[idx])
    return train_scores, test_scores, train_time, test_time, estimators

def learning_curve_func(clf, X_train, y_train, scoring='accuracy', cv=5, verbose=False, random_state=100, permu=False):
    #print("here")
    if permu:        
        np.random.seed(random_state)
        new_list = np.random.permutation(len(X_train))
        if type(X_train) == np.ndarray:
            p_X_train = X_train[new_list]
            p_y_train = y_train[new_list]
        else:
            p_X_train = X_train.to_numpy()[new_list]
            p_y_train = y_train.to_numpy()[new_list]
    else:
        p_X_train = X_train
        p_y_train = y_train
        np.random.seed(random_state)
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    train_sizes = np.arange(.1, 1.1, 0.1)
    scores = learning_curve(clf, p_X_train, p_y_train, cv=kfold, train_sizes = train_sizes, scoring=scoring, return_times=True)
    return (train_sizes*100).astype(np.int32), np.array(scores[1]).mean(axis=1), np.array(scores[2]).mean(axis=1) #\
        #, np.array(scores[2]).mean(axis=1), np.array(scores[3]).mean(axis=1)

def cal_chi2_scores(X_train, y_train):
    chi2_scores = chi2(X_train, y_train)
    chi2_sorted = np.argsort(-chi2_scores[0])
    for i in chi2_sorted:
        print(X_train.columns[i], chi2_scores[0][i])  

