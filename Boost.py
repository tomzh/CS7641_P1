import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from Utils import *

def Boost_model_selection_0(X, y, X_train, X_test, y_train, y_test, cv=8, ds_flag=0, verbose=False):    
    seed = 10
    np.random.seed(seed)
    p_clf = DecisionTreeClassifier(random_state=0,class_weight='balanced')
    path = p_clf.cost_complexity_pruning_path(X_train, y_train)
    org_ccp_alphas, impurities = path.ccp_alphas, path.impurities
    org_ccp_alphas = org_ccp_alphas[:-1]
    impurities = impurities[:-1]
    
    scoring = 'accuracy'
    scoring_title = 'Accuracy'
    
    init_args2 = {'class_weight':'balanced', 'criterion':'gini'}
    init_args1 = {'n_estimators':50, 'algorithm':"SAMME.R", 'learning_rate': 0.15}
    
    alpha_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, init_args1, 'ccp_alpha', org_ccp_alphas, 
                                      scoring, cv=cv, random_state=seed,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=1)
    
    np.random.seed(seed)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini','ccp_alpha':0.0503}
    lr_args={'algorithm':"SAMME.R", 'n_estimators':100} 
    scoring='accuracy'
    lrs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 1]
    lr_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, lr_args, 'learning_rate', lrs, 
                                      scoring, cv=cv, random_state=seed,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=0)
    
    np.random.seed(seed)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini','ccp_alpha':0.0503}
    iters_args={'algorithm':"SAMME.R", 'learning_rate': 0.15}
    iters = [3, 6, 10, 15, 20, 30, 40, 60, 80, 100]
    iters_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, iters_args, 'n_estimators', iters, 
                                      scoring, cv=cv, random_state=seed,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=0)

    comp_prunning = [
                  {'X': org_ccp_alphas, 'Y':[alpha_scores[0], alpha_scores[1]], 
                 'title': f'Alpha vs {scoring_title}\n(Iterations=50, lr=0.15)', 
                'labels':['Training Score','Validate Score'],
                 'axis_lablels':['Alpha', f'{scoring_title}']},
                 {'X':lrs, 'Y':[lr_scores[0], lr_scores[1]], 
                   'title': f'Learning Rate vs {scoring_title}\n(alpha=0.0503, Iterations=50)', 
                  'labels':['Training Score','Validate Score'],
                   'axis_lablels':['Learning Rate', f'{scoring_title}']},
                 {'X':iters, 'Y':[iters_scores[0], iters_scores[1]], 
                   'title': f'Train Iterations vs {scoring_title}\n(alpha=0.0503, lr=0.15)', 
                  'labels':['Training Score','Validate Score'],
                   'axis_lablels':['Max Iterations', f'{scoring_title}']},
                ]
    plat_subgraph(comp_prunning, 8+ds_flag)
    
    print(f"Avg Training Time: {np.array(alpha_scores[2]).mean()}, Avg Validation Time: {np.array(alpha_scores[3]).mean()}")

def Boost_model_selection_1(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=1, verbose=False):
    np.random.seed(100)
    p_clf = DecisionTreeClassifier(random_state=0,class_weight='balanced')
    path = p_clf.cost_complexity_pruning_path(X_train, y_train)
    org_ccp_alphas, impurities = path.ccp_alphas, path.impurities
    org_ccp_alphas = org_ccp_alphas[:-1]
    impurities = impurities[:-1]
    
    scoring='recall'
    scoring_title = 'Recall'
    np.random.seed(100)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini'}
    args = {'n_estimators':100,
           'learning_rate': 0.05}
    ccp_alphas = org_ccp_alphas[-20:]
    alpha_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, args, 'ccp_alpha', ccp_alphas, 
                                      scoring, cv=cv, random_state=100,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=1)
    
    
    np.random.seed(100)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini','ccp_alpha':0.009}
    args={'n_estimators':100} 
    lrs = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
    lrs_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, args, 'learning_rate', lrs, 
                                      scoring, cv=cv, random_state=100,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=0)
    
    
    np.random.seed(100)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini','ccp_alpha':0.009}
    args={'learning_rate': 0.05}
    iters = [50, 100, 200, 300, 400, 500]
    iters_scores = cross_validate_func(AdaBoostClassifier, X_train, y_train, args, 'n_estimators', iters, 
                                      scoring, cv=cv, random_state=100,
                                      learner2=DecisionTreeClassifier, init_args2=init_args2, arg_test=0)
    
    comp_prunning = [
                  {'X': ccp_alphas, 'Y':[alpha_scores[0], alpha_scores[1]], 
                 'title': f'Alpha vs {scoring_title}\n(Iterations=100, lr=0.05)', 
                'labels':['Training Score','Validate Score'],
                 'axis_lablels':['Alpha', f'{scoring_title}']},
                 {'X':lrs, 'Y':[lrs_scores[0], lrs_scores[1]], 
                   'title': f'Learning Rate vs {scoring_title}\n(alpha=0.009, Iterations=100)', 
                  'labels':['Training Score','Validate Score'],
                   'axis_lablels':['Learning Rate', f'{scoring_title}']},
                 {'X':iters, 'Y':[iters_scores[0], iters_scores[1]], 
                   'title': f'Train Iterations vs {scoring_title}\n(alpha=0.009, lr=0.05)', 
                  'labels':['Training Score','Validate Score'],
                   'axis_lablels':['Max Iterations', f'{scoring_title}']},
                ]
    plat_subgraph(comp_prunning, 8+ds_flag)
    print(f"Avg Training Time: {np.array(alpha_scores[2]).mean()}, Avg Validation Time: {np.array(alpha_scores[3]).mean()}")

def Boost_learning_curve(hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test,
                      bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test,
                      cv=5, verbose=False):
    
    hd_seed=10
    np.random.seed(hd_seed)
    init_args2 = {'class_weight':'balanced', 'criterion':'gini', 'ccp_alpha':0.0503}
    init_args1 = {'n_estimators':10, 'learning_rate': 0.15}
    
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, **init_args2),**init_args1)
    hd_size_ratio_plot, hd_sample_size_train_scroes, hd_sample_size_test_scroes = \
        learning_curve_func(clf, hd_X_train, hd_y_train, scoring='accuracy', cv=8, verbose=False, random_state=hd_seed)
        
    comp_final_test(AdaBoostClassifier(DecisionTreeClassifier()), 
                    AdaBoostClassifier(DecisionTreeClassifier(random_state=0, **init_args2),**init_args1),
                    hd_X_train, hd_y_train, hd_X_test,  hd_y_test, score_func=accuracy_score)

    np.random.seed(100)
    init_args2 = ({'ccp_alpha':0.009, 'class_weight':'balanced'})
    init_args1 = ({'n_estimators':100, 'learning_rate': 0.05})    
    bank_clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, **init_args2),**init_args1)
    bank_size_ratio_plot, bank_sample_size_train_scroes, bank_sample_size_test_scroes = \
        learning_curve_func(bank_clf, bank_X_train, bank_y_train, scoring='recall', cv=5, verbose=False, random_state=100)
    
    comp_final_test(AdaBoostClassifier(DecisionTreeClassifier()), 
                    AdaBoostClassifier(DecisionTreeClassifier(random_state=0, **init_args2),**init_args1), 
                    bank_X_train, bank_y_train, bank_X_test, bank_y_test, score_func=recall_score)
        
    
    comp_prunning = [
              {'X': hd_size_ratio_plot, 'Y':[hd_sample_size_train_scroes, hd_sample_size_test_scroes], 
             'title': 'Training Sample Size vs Accuracy (Heart Disease DS)', 
            'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Accuracy']},
             {'X':bank_size_ratio_plot, 'Y':[bank_sample_size_train_scroes, bank_sample_size_test_scroes], 
               'title': 'Training Sample Size vs Recall (Credit Card DS)', 
              'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Recall']},
            ]
    plat_subgraph(comp_prunning, 10)