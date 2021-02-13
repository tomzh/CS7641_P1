import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from Utils import *

def SVM_model_selection_0(X, y, X_train, X_test, y_train, y_test, cv=8, ds_flag=0):
    seed=10
    np.random.seed(seed)
    k_list = ['linear', 'poly', 'rbf', 'sigmoid']
    C_list = [0.01, 0.03, 0.05, 0.1, 0.3,0.5,0.7,0.9,1]
    scoring='accuracy'
    
    all_scores = []
    for k in k_list:
        args = {'gamma':'auto', 'kernel':k}
        scores = cross_validate_func(SVC, X_train, y_train, args, 'C', C_list, 
                                      scoring, cv=cv, random_state=seed,scale_data=True)
        all_scores.append(scores)
    
    plot_data(C_list, [all_scores[0][0], all_scores[0][1],all_scores[1][0], all_scores[1][1],
                      all_scores[2][0], all_scores[2][1],all_scores[3][0], all_scores[3][1]],
              'C Value', 'Accuracy', 'a. C Value vs Accuracy Score with Different Kernels\n(Heart Disease DS)', 
              ['Training Score: linear','Val. Score: linear','Training Score: poly','Val. Score: poly',
               'Training Score: rbf','Val. Score: rbf','Training Score: sigmoid','Val. Score: sigmoid'], 11, sub='a')
    print(f"Avg Training Time: {np.array(all_scores[0][2]).mean()}, Avg Validation Time: {np.array(all_scores[0][3]).mean()}")
    
def SVM_model_selection_1(X, y, X_train, X_test, y_train, y_test, cv=5, ds_flag=1):
    seed=100
    np.random.seed(seed)
    k_list = ['linear', 'poly', 'rbf', 'sigmoid']
    C_list = [0.01, 0.03, 0.05, 0.1, 0.3,0.5,0.7,0.9,1]
    scoring='recall'
    
    all_scores = []
    for k in k_list:
        args = {'gamma':'auto', 'kernel':k, 'class_weight':'balanced'}
        scores = cross_validate_func(SVC, X_train, y_train, args, 'C', C_list, 
                                      scoring, cv=cv, random_state=seed,scale_data=True)
        all_scores.append(scores)
    
    plot_data(C_list, [all_scores[0][0], all_scores[0][1],all_scores[1][0], all_scores[1][1],
                      all_scores[2][0], all_scores[2][1],all_scores[3][0], all_scores[3][1]],
              'C Value', 'Recall', 'b. C Value vs Recall Score with Different Kernels\n(Credit Card DS)', 
              ['Training Score: linear','Val. Score: linear','Training Score: poly','Val. Score: poly',
               'Training Score: rbf','Val. Score: rbf','Training Score: sigmoid','Val. Score: sigmoid'], 11, sub='b')
    print(f"Avg Training Time: {np.array(all_scores[0][2]).mean()}, Avg Validation Time: {np.array(all_scores[0][3]).mean()}")
    

def SVM_learning_curve(hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test,
                      bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, verbose=False):
    hd_seed = 10
    np.random.seed(hd_seed)
    h_best_arg = {'gamma':'auto',  'kernel':'rbf', 'C':.5}
    
    clf = make_pipeline(StandardScaler(), SVC(**h_best_arg))
    hd_size_ratio_plot, hd_sample_size_train_scroes, hd_sample_size_test_scroes = \
        learning_curve_func(clf, hd_X_train, hd_y_train, scoring='accuracy', cv=10, verbose=False, random_state=hd_seed)   
        
    comp_final_test(make_pipeline(StandardScaler(), SVC()), make_pipeline(StandardScaler(), SVC(**h_best_arg)),
                    hd_X_train, hd_y_train, hd_X_test,  hd_y_test, score_func=accuracy_score)
    
    np.random.seed(100)
    bank_best_arg = {'gamma':'auto', 'class_weight':'balanced','kernel':'rbf', 'C':0.7} 
    bank_clf = make_pipeline(StandardScaler(), SVC(**bank_best_arg))
    bank_size_ratio_plot, bank_sample_size_train_scroes, bank_sample_size_test_scroes = \
        learning_curve_func(bank_clf, bank_X_train, bank_y_train, scoring='recall', cv=5, verbose=False, random_state=100)
    
    comp_final_test(make_pipeline(StandardScaler(), SVC()), make_pipeline(StandardScaler(), SVC(**bank_best_arg)), 
                    bank_X_train, bank_y_train, bank_X_test, bank_y_test, score_func=recall_score)
    
    comp_prunning = [
              {'X': hd_size_ratio_plot, 'Y':[hd_sample_size_train_scroes, hd_sample_size_test_scroes], 
             'title': 'Training Sample Size vs Accuracy (Heart Disease DS)', 
            'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Accuracy']},
             {'X':bank_size_ratio_plot, 'Y':[bank_sample_size_train_scroes, bank_sample_size_test_scroes], 
               'title': 'Training Sample Size vs Recall (Credit Card DS)', 
              'labels':['Training Score','Validate Score'], 'axis_lablels':['Training Sample Size (%)', 'Recall']},
            ]
    plat_subgraph(comp_prunning, 12)