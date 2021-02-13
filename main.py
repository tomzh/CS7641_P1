from sklearn.model_selection import train_test_split
import warnings
from time import time
from DT import *
from Boost import *
from NN import *
from SVM import *
from KNN import *
from Utils import *
   
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #Prepare datasets
    bank_dataset = prepare_bank_dataset()
    plot_hist(bank_dataset.iloc[:, 0], '', 'Frequency', 'b. Credit Card Dataset Class Histogram', file_name='figure_1b',
              lables=['non-churned','churned'])
    bank_train = bank_dataset.iloc[:, 1:]
    bank_test = bank_dataset.iloc[:, 0]
    bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test = \
        train_test_split(bank_train, bank_test, test_size=.3, random_state=10, shuffle=True, stratify=bank_test) 
    
    #print("\nCredit Card Customer Dataset Chi2 Scores:")  
    #cal_chi2_scores(bank_X_train, bank_y_train)      
    hd_dataset = load_dataset("datasets/heart_disease_dataset.csv")
    plot_hist(hd_dataset.iloc[:, -1], '', 'Frequency', 'a. Heart Disease Dataset Class Histogram', file_name='figure_1a',
              lables=['no disease','disease'])
    hd_train = hd_dataset.iloc[:, :-1]
    hd_test = hd_dataset.iloc[:, -1]
    hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test = \
        train_test_split(hd_train, hd_test, test_size=.3, random_state=10, shuffle=True, stratify=hd_test)
        
    time_s = time()    
    #'''
    print("\nDecision Tree results:")        
    DT_model_selection(hd_train, hd_test, hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test, cv=8, ds_flag=0)
    DT_model_selection(bank_train, bank_test, bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, ds_flag=1)
    
    DT_learning_curve(hd_X_train, hd_X_test, hd_y_train, hd_y_test,
                     bank_X_train,  bank_X_test, bank_y_train, bank_y_test, cv=5, verbose=False)
    print("Elapsed time:", time() - time_s)
    time_s = time()
    
    print("\nNeural Network results:")  
    NN_model_selection_0(hd_train, hd_test, hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test, cv=8, ds_flag=0)
    NN_model_selection_1(bank_train, bank_test, bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, ds_flag=1)
    NN_learning_curve(hd_X_train, hd_X_test, hd_y_train, hd_y_test,
                      bank_X_train,  bank_X_test, bank_y_train, bank_y_test, cv=5, verbose=False)
    print("Elapsed time:", time() - time_s)
    
    time_s = time()
    print("\nBoost results:") 
    Boost_model_selection_0(hd_train, hd_test, hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test, cv=8, ds_flag=0)
    Boost_model_selection_1(bank_train, bank_test, bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, ds_flag=1)
    
    Boost_learning_curve(hd_X_train, hd_X_test, hd_y_train, hd_y_test,
                      bank_X_train,  bank_X_test, bank_y_train, bank_y_test, cv=5, verbose=False)
    print("Elapsed time:", time() - time_s)
    
    time_s = time()
    print("\nSVM results:") 
    SVM_model_selection_0(hd_train, hd_test, hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test, cv=8, ds_flag=0)
    SVM_model_selection_1(bank_train, bank_test, bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, ds_flag=1)
    SVM_learning_curve(hd_X_train, hd_X_test, hd_y_train, hd_y_test,
                      bank_X_train,  bank_X_test, bank_y_train, bank_y_test, cv=5, verbose=False)
    print("Elapsed time:", time() - time_s)
    
    time_s = time()
    print("\nKNN results:") 
    KNN_model_selection_0(hd_train, hd_test, hd_X_train,  hd_X_test,  hd_y_train,  hd_y_test, cv=8, ds_flag=0)
    KNN_model_selection_1(bank_train, bank_test, bank_X_train,  bank_X_test,  bank_y_train,  bank_y_test, cv=5, ds_flag=1)
    
    KNN_learning_curve(hd_X_train, hd_X_test, hd_y_train, hd_y_test,
                      bank_X_train,  bank_X_test, bank_y_train, bank_y_test, cv=5, verbose=False)
    
    print("Elapsed time:", time() - time_s)
    
    
    
    
