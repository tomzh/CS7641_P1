import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from Utils import *


def KNN_model_selection_0(X, y, X_train, X_test, y_train, y_test, cv=8, ds_flag=0): 
    seed=10
    cv = 10
    np.random.seed(seed)
    
    k_list = np.arange(1,21,1)
    p_list = [1, 2]
    scoring='accuracy'
    all_scores = []
    for p in p_list:
        args = {'weights':'distance', 'p':p}
        k_scores = cross_validate_func(KNeighborsClassifier, X_train, y_train, args, 'n_neighbors', k_list, 
                                      scoring, cv=cv, random_state=seed,scale_data=True)
        all_scores.append(k_scores)
    
    plot_data(k_list, [all_scores[0][0], all_scores[0][1],all_scores[1][0], all_scores[1][1]],
          'Neighbors (K)', 'Accuracy', 'Neighbors vs Accuracy Score with Different Distance', 
          ['Training Score: Manhattan dist.','Val. Score: Manhattan dist.',
           'Training Score: Euclidean dist.','Val. Score: Euclidean dist.'], 13)
    print(f"Avg Training Time: {np.array(all_scores[0][2]).mean()}, Avg Validation Time: {np.array(all_scores[0][3]).mean()}")

def KNN_model_selection_1(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=1):
    seed=100
    np.random.seed(seed)
    down_X_train, down_y_train = resample_Xy(X_train.to_numpy(), y_train.to_numpy(), up_sample=False, apply=True)
    k_list = np.arange(1,21,1)
    p_list = [1, 2]
    scoring='recall'
    org_all_scores = []
    down_all_scores = []
    for p in p_list:
        args = {'weights':'distance','p':p}
        #print(k_list)
        org_scores = cross_validate_func(KNeighborsClassifier, X_train, y_train, args, 'n_neighbors', k_list, 
                                      scoring, cv=cv, random_state=seed,scale_data=True)
        org_all_scores.append(org_scores)
        down_scores = cross_validate_func(KNeighborsClassifier,down_X_train, down_y_train, args, 'n_neighbors', k_list, 
                                      scoring, cv=cv, random_state=seed,scale_data=True)
        down_all_scores.append(down_scores)
    
    comp_prunning = [
              {'X': k_list, 
               'Y':[org_all_scores[0][0], org_all_scores[0][1],org_all_scores[1][0], org_all_scores[1][1]], 
             'title': f'Neighbors vs Recall Score (original train set)', 
            'labels':['Train Score: Manhattan dist.','Val. Score: Manhattan dist.', 'Train Score: Euclidean dist.',
                      'Val. Score: Euclidean dist.'],
             'axis_lablels':['Neighbors (K)', 'Recall']},
              {'X': k_list, 
               'Y':[down_all_scores[0][0], down_all_scores[0][1],down_all_scores[1][0], down_all_scores[1][1]], 
             'title': f'Neighbors vs Recall Score (downsample train set)', 
            'labels':['Train Score: Manhattan dist.','Val. Score: Manhattan dist.', 'Train Score: Euclidean dist.',
                      'Val. Score: Euclidean dist.'],
             'axis_lablels':['Neighbors (K)', 'Recall']},
            ]
    
    plat_subgraph(comp_prunning, 13+ds_flag)
    print(f"Avg Training Time: {np.array(down_all_scores[0][2]).mean()}, Avg Validation Time: {np.array(down_all_scores[0][3]).mean()}")

def KNN_learning_curve(hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test,
                      bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, verbose=False):
    hd_seed = 10
    np.random.seed(hd_seed)
    h_best_arg = {'weights':'distance',  'n_neighbors':8, 'p':2}
    #new_list = np.random.permutation(len(hd_X_train))
    #hd_p_X_train = hd_X_train.to_numpy()[new_list]
    #hd_p_y_train = hd_y_train.to_numpy()[new_list]
        
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(**h_best_arg))
    hd_size_ratio_plot, hd_sample_size_train_scroes, hd_sample_size_test_scroes = \
        learning_curve_func(clf, hd_X_train, hd_y_train, scoring='accuracy', cv=10, verbose=False, random_state=hd_seed, permu=True)

    comp_final_test(make_pipeline(StandardScaler(), KNeighborsClassifier()), make_pipeline(StandardScaler(), KNeighborsClassifier(**h_best_arg)),
                    hd_X_train, hd_y_train, hd_X_test,  hd_y_test, score_func=accuracy_score)
    
    bank_seed=10
    np.random.seed(bank_seed)
    down_X_train, down_y_train = resample_Xy(bank_X_train.to_numpy(), bank_y_train.to_numpy(), up_sample=False, apply=True)
    np.random.seed(bank_seed)
    bank_best_arg = {'weights':'distance','p':1, 'n_neighbors':14}
    #new_list = np.random.permutation(len(down_X_train))
    #p_X_train = down_X_train[new_list]
    #p_y_train = down_y_train[new_list] 
    bank_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(**bank_best_arg))
    bank_size_ratio_plot, bank_sample_size_train_scroes, bank_sample_size_test_scroes = \
        learning_curve_func(bank_clf, down_X_train, down_y_train, scoring='recall', cv=5, verbose=False, random_state=bank_seed, permu=True)
        
    comp_final_test(make_pipeline(StandardScaler(), KNeighborsClassifier()), make_pipeline(StandardScaler(), KNeighborsClassifier(**bank_best_arg)),
                    bank_X_train, bank_y_train, bank_X_test, bank_y_test, score_func=recall_score, down_sample=True,
                    down_X=down_X_train, down_y=down_y_train)
    
    comp_prunning = [
              {'X': hd_size_ratio_plot, 'Y':[hd_sample_size_train_scroes, hd_sample_size_test_scroes], 
             'title': 'Training Sample Size vs Accuracy (Heart Disease DS)', 
            'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Accuracy']},
             {'X':bank_size_ratio_plot, 'Y':[bank_sample_size_train_scroes, bank_sample_size_test_scroes], 
               'title': 'Training Sample Size vs Recall (Credit Card DS)', 
              'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Recall']},
            ]
    plat_subgraph(comp_prunning, 15)
    
    
    
    