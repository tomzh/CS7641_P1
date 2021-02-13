import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from Utils import *

 
def DT_learning_curve(hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test,
                      bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test,
                      cv=5, verbose=False):
    
    np.random.seed(100)    
    #Heart DS
    best_arg = {'ccp_alpha':0.01391, 'class_weight':'balanced', 'criterion':'gini'}
    clf = DecisionTreeClassifier(**best_arg)
    hd_size_ratio_plot, hd_sample_size_train_scroes, hd_sample_size_test_scroes = \
        learning_curve_func(clf, hd_X_train, hd_y_train, scoring='accuracy', cv=5, verbose=False, random_state=100, permu=False)
    
    comp_final_test(DecisionTreeClassifier(), DecisionTreeClassifier(**best_arg),
                    hd_X_train, hd_y_train, hd_X_test,  hd_y_test, score_func=accuracy_score)
    
    ##Credit Card DS
    best_arg = {'ccp_alpha':0.00146, 'class_weight':'balanced', 'criterion':'gini'}
    np.random.seed(100)
    clf = DecisionTreeClassifier(**best_arg)
    bank_size_ratio_plot, bank_sample_size_train_scroes, bank_sample_size_test_scroes = \
        learning_curve_func(clf, bank_X_train,  bank_y_train, scoring='recall', cv=5, verbose=False, random_state=100, permu=False)
    
    comp_final_test(DecisionTreeClassifier(), DecisionTreeClassifier(**best_arg), 
                    bank_X_train, bank_y_train, bank_X_test, bank_y_test, score_func=recall_score)
        
    comp_prunning = [
              {'X': hd_size_ratio_plot, 'Y':[hd_sample_size_train_scroes, hd_sample_size_test_scroes], 
             'title': 'Training Sample Size vs Accuracy (Heart Disease DS)', 
            'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Accuracy']},
             {'X':bank_size_ratio_plot, 'Y':[bank_sample_size_train_scroes, bank_sample_size_test_scroes], 
               'title': 'Training Sample Size vs Recall (Credit Card DS)', 
              'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Recall']},
            ]
    plat_subgraph(comp_prunning, 4)
    
def DT_model_selection(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=0, verbose=False):
    np.random.seed(100)
    p_clf = DecisionTreeClassifier(random_state=0,class_weight='balanced')
    path = p_clf.cost_complexity_pruning_path(X, y)
    org_ccp_alphas, impurities = path.ccp_alphas, path.impurities
    org_ccp_alphas = org_ccp_alphas[:-1]
    impurities = impurities[:-1]
    if ds_flag == 1:
        org_ccp_alphas = org_ccp_alphas[120:]
        org_ccp_alphas = org_ccp_alphas[org_ccp_alphas < 0.01]
        scoring = 'recall'
        scoring_title = 'Recall'
    else:
        scoring = 'accuracy'
        scoring_title = 'Accuracy'
    
    init_args = {'class_weight':'balanced', 'criterion':'gini'}
    np.random.seed(100)
    alpha_scores = cross_validate_func(DecisionTreeClassifier, X_train, y_train,
                    init_args, 'ccp_alpha', org_ccp_alphas, scoring, cv=cv, verbose=False, random_state=100)
    np.random.seed(100)
    alpha_en_scores = cross_validate_func(DecisionTreeClassifier, X_train, y_train,
                    {'class_weight':'balanced', 'criterion':'entropy'}, 
                                   'ccp_alpha', org_ccp_alphas, scoring, cv=cv, verbose=False, random_state=100)    
    init_args = {'class_weight':'balanced', 'criterion':'gini'}
    np.random.seed(100)
    depth_scores = cross_validate_func(DecisionTreeClassifier, X_train, y_train,
                    init_args, 'max_depth', np.arange(1, 20, 1), scoring, cv=cv, verbose=False, random_state=100)
    np.random.seed(100)
    depth_en_scores = cross_validate_func(DecisionTreeClassifier, X_train, y_train,
                    {'class_weight':'balanced', 'criterion':'entropy'}, 
                                   'max_depth', np.arange(1, 20, 1), scoring, cv=cv, verbose=False, random_state=100)
    
    comp_prunning = [
                  {'X': np.arange(1, 20, 1), 'Y':[depth_scores[0], depth_scores[1], depth_en_scores[0], depth_en_scores[1]], 
                 'title': f'Max_depth vs {scoring_title}', 
                'labels':['Training Score: Gini','Validate Score: Gini', 'Training Score: Entropy','Validate Score: Entropy'],
                 'axis_lablels':['Max_depth (Pre-prunning)', f'{scoring_title}']},
                 {'X':org_ccp_alphas, 'Y':[alpha_scores[0], alpha_scores[1], alpha_en_scores[0], alpha_en_scores[1]], 
                   'title': f'Alpha vs {scoring_title}', 
                  'labels':['Training Score: Gini','Validate Score: Gini', 'Training Score: Entropy','Validate Score: Entropy'],
                   'axis_lablels':['Alpha (Post-prunning)', f'{scoring_title}']},
                ]
    plat_subgraph(comp_prunning, 2+ds_flag)
    
    print(f"Avg Training Time: {np.array(alpha_scores[2]).mean()}, Avg Validation Time: {np.array(alpha_scores[3]).mean()}")
    
    if verbose:
        max_i = np.array(alpha_scores[1]).argmax()
        print("Best arg:", org_ccp_alphas[max_i], alpha_scores[1][max_i])
