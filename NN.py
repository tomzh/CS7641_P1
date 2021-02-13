import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from Utils import *


def NN_model_selection_0(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=0, verbose=False):    
    if ds_flag == 1:
        scoring = 'recall'
        scoring_title = 'Recall'
    else:
        scoring = 'accuracy'
        scoring_title = 'Accuracy'
    
    hd_seed = 9
    np.random.seed(hd_seed)
    max_tier=200
    init_args = {'max_iter':max_tier, 'activation':'relu', 'solver': 'adam', 'random_state':hd_seed}
    args = [(10,),(30,), (50,), (80,), (100,),(120,),(150,),(180,), (200,)]
    args_plot = [u[0] for u in args]
    scores1 = cross_validate_func(MLPClassifier, X_train, y_train,
                        init_args, 'hidden_layer_sizes', args, 'accuracy', cv=cv, scale_data=True, random_state=hd_seed)
    np.random.seed(hd_seed)
    init_args2 = {'max_iter':max_tier, 'activation':'relu', 'solver': 'sgd', 'random_state':hd_seed}
    scores2 = cross_validate_func(MLPClassifier, X_train, y_train,
                        init_args2, 'hidden_layer_sizes', args, 'accuracy', cv=cv, scale_data=True, random_state=hd_seed)
    
    np.random.seed(hd_seed)
    init_args3 = {'max_iter':max_tier, 'activation':'logistic', 'solver': 'lbfgs', 'random_state':hd_seed}
    scores3 = cross_validate_func(MLPClassifier, X_train, y_train,
                        init_args3, 'hidden_layer_sizes', args, 'accuracy', cv=cv, scale_data=True, random_state=hd_seed)
    
    
    np.random.seed(hd_seed)
    init_args4 = {'max_iter':max_tier, 'activation':'relu', 'solver': 'sgd', 'hidden_layer_sizes':180, 'random_state':hd_seed}
    args4 = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    scores4 = cross_validate_func(MLPClassifier, X_train, y_train,
                        init_args2, 'max_iter', args4, 'accuracy', cv=cv, scale_data=True, random_state=hd_seed)
    
    comp_prunning = [
              {'X': args_plot, 
               'Y':[scores1[0], scores1[1], scores2[0], scores2[1], scores3[0], scores3[1]], 
             'title': f'Units vs Accuracy', 
            'labels':['Train Score: adam(Relu)','Val. Score: adam(Relu)', 'Train Score: sgd(Relu)',
                      'Val. Score: sgd(Relu)', 'Train Score: lbfgs(Log.)','Val. Score: lbfgs(Log.)'],
             'axis_lablels':['Units', 'Accuracy']},
              {'X': args4, 
               'Y':[scores4[0], scores4[1]], 
             'title': f'Max iterations vs Accuracy: (sgd, Relu, Units=180)', 
            'labels':['Training Score','Validate Score'],
             'axis_lablels':['Max iterations', 'Accuracy']},
            ]
    
    plat_subgraph(comp_prunning, 5+ds_flag)
    
    print(f"Avg Training Time: {np.array(scores4[2]).mean()}, Avg Validation Time: {np.array(scores4[3]).mean()}")

def NN_model_selection_1(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=1, verbose=False):    
    max_tier=200
    seed = 100
    np.random.seed(seed)
    down_X_train, down_y_train = resample_Xy(X_train.to_numpy(), y_train.to_numpy(), up_sample=False, apply=True)
    args = [(6,), (8,), (10,), (20,), (30,), (40,), (60,), (80,), (100,)]
    args_plot = [u[0] for u in args]
    sovers = ['adam', 'sgd', 'lbfgs']
    org_sover_scores = []
    down_sover_scores = []
    for sover in sovers:
        if sover == 'lbfgs':
            init_args = {'max_iter':max_tier, 'activation':'logistic', 'solver': sover, 'random_state':seed}
        else:        
            init_args = {'max_iter':max_tier, 'activation':'relu', 'solver': sover, 'random_state':seed}
        scores1 = cross_validate_func(MLPClassifier, X_train, y_train,
                        init_args, 'hidden_layer_sizes', args, 'recall', cv=cv, scale_data=True, random_state=seed)
        org_sover_scores.append(scores1)
        scores2 = cross_validate_func(MLPClassifier, down_X_train, down_y_train,
                        init_args, 'hidden_layer_sizes', args, 'recall', cv=cv, scale_data=True, random_state=seed)
        down_sover_scores.append(scores2)
    
    lb_args = [50, 80, 100, 150, 200, 250, 300]
    h_layers = [(40,), (20,20)]   
    
    layer_scores = []
    for layer in h_layers:
        lb_scores1 = cross_validate_func(MLPClassifier, down_X_train, down_y_train,
                        {'activation':'relu', 'solver': 'adam','hidden_layer_sizes':layer, 'random_state':100}, 
                        'max_iter', lb_args, 'recall', cv=cv, scale_data=True, random_state=100) 
        layer_scores.append(lb_scores1)
    
    comp_prunning = [
              {'X': args_plot, 
               'Y':[org_sover_scores[0][0], org_sover_scores[0][1], org_sover_scores[1][0], org_sover_scores[1][1], 
                    org_sover_scores[2][0], org_sover_scores[2][1]], 
             'title': f'Units vs Recall\n(original train set)', 
             'labels':['Train Score: adam(Relu)','Val. Score: adam(Relu)', 'Train Score: sgd(Relu)',
                      'Val. Score: sgd(Relu)', 'Train Score: lbfgs(Log.)','Val. Score: lbfgs(Log.)'],
             'axis_lablels':['Units', 'Recall']},
              {'X': args_plot, 
               'Y':[down_sover_scores[0][0], down_sover_scores[0][1], down_sover_scores[1][0], down_sover_scores[1][1], 
                    down_sover_scores[2][0], down_sover_scores[2][1]], 
             'title': f'Units vs Recall\n(downsample train set)', 
             'labels':['Train Score: adam(Relu)','Val. Score: adam(Relu)', 'Train Score: sgd(Relu)',
                      'Val. Score: sgd(Relu)', 'Train Score: lbfgs(Log.)','Val. Score: lbfgs(Log.)'],
             'axis_lablels':['Units', 'Recall']},              
              {'X': lb_args, 
               'Y':[layer_scores[0][0], layer_scores[0][1], layer_scores[1][0], layer_scores[1][1]], 
             'title': f'Max Iterations vs Recall\n(adam, downsample)', 
             'labels':['Train Score: H_size=(40, )','Val. Score: H_size=(40, )', 'Train Score: H_size=(20, 20)',
                      'Val. Score: H_size=(20, 20)'],
             'axis_lablels':['Max Iterations', 'Recall']},
            ]
    
    plat_subgraph(comp_prunning, 5+ds_flag)
    
    print(f"Avg Training Time: {np.array(org_sover_scores[0][2]).mean()}, Avg Validation Time: {np.array(org_sover_scores[0][3]).mean()}")

def NN_learning_curve(hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test,
                      bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test,
                      cv=5, verbose=False):
    hd_seed = 9
    np.random.seed(hd_seed)
    h_best_arg = {'max_iter':270, 'hidden_layer_sizes': (180,), 'activation':'relu', 'solver': 'sgd', 'random_state':hd_seed}
    
    clf = make_pipeline(StandardScaler(), MLPClassifier(**h_best_arg))
    hd_size_ratio_plot, hd_sample_size_train_scroes, hd_sample_size_test_scroes = \
        learning_curve_func(clf, hd_X_train, hd_y_train, scoring='accuracy', cv=8, verbose=False, random_state=hd_seed, permu=False)
    
    comp_final_test(make_pipeline(StandardScaler(), MLPClassifier(random_state=hd_seed)),
                    make_pipeline(StandardScaler(), MLPClassifier(**h_best_arg)),
                    hd_X_train, hd_y_train, hd_X_test,  hd_y_test, score_func=accuracy_score)

    bank_seed=100
    np.random.seed(bank_seed)
    down_X_train, down_y_train = resample_Xy(bank_X_train.to_numpy(), bank_y_train.to_numpy(), up_sample=False, apply=True)
    bank_best_arg = {'max_iter':90, 'hidden_layer_sizes': (20,20), 'activation':'relu', 'solver': 'adam', 'random_state':bank_seed}    
    bank_clf = make_pipeline(StandardScaler(), MLPClassifier(**bank_best_arg))
    bank_size_ratio_plot, bank_sample_size_train_scroes, bank_sample_size_test_scroes = \
        learning_curve_func(bank_clf, down_X_train, down_y_train, scoring='recall', cv=5, verbose=False, random_state=bank_seed, permu=True)
    
    
    comp_final_test(make_pipeline(StandardScaler(), MLPClassifier(random_state=bank_seed)), 
                    make_pipeline(StandardScaler(), MLPClassifier(**bank_best_arg)),
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
    plat_subgraph(comp_prunning, 7)
    
    