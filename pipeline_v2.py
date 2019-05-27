# Based on https://github.com/rayidghani/magicloops

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score as accuracy

from sklearn.metrics import *

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.metrics import precision_recall_curve

pd.options.mode.chained_assignment = None  # default='warn'

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def read_csv(url):
  print ("Reading file...")
  df= pd.read_csv(url) 
  print ("Done")
  return df

# https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32
def detect_outlier(data):
    
    outliers=[]
    threshold=4
    mean_1 = np.mean(data)
    std_1 =np.std(data)    
    
    for y in data:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

def explore_data(df, selected_variables, var_for_corr):
  print ("Data exploration...\n")
  
  n_rows = df.shape[0]
  print("Number of rows: "+str(n_rows)+"\n")

  print("Columns and types of data:")
  print(df.dtypes)
  print("\n")

  print("Statistics for selected variables:")

  for variable in selected_variables:
    print(df[variable].describe())
    print("Number of outliers (>4 standard dev):"+str(len(detect_outlier(df[variable]))))
    df.hist(column=variable)
    print("\n")

  print("\n")
  print("Correlation between "+str(var_for_corr))

  print(df[var_for_corr[0]].corr(df[var_for_corr[1]]))

#Filling in missing values with mean
def pre_process_data(df, columns_to_process):
  print ("Pre processing data...")

  #Get average of each column
  averages={}
  for column in columns_to_process:
    averages[column] = df[column].mean()


  #Replace missing values with averages
  for index, row in df.iterrows():
    for column in columns_to_process:
      if(pd.isna(row[column])):
        #this line could be tider
        df.iloc[index,df.columns.get_loc(column)]=averages[column]

  for index, row in df.iterrows():
    for column in columns_to_process:
      if(pd.isna(row[column])):
        print("error")

  print ("Done")
  return df


def create_dummies(df, cols_to_transform):
  return pd.get_dummies(df, dummy_na=True, columns = cols_to_transform, drop_first=True)

def create_temp_validation_train_and_testing_sets(df, features, data_column, label_column, split_thresholds, test_window, gap_training_test):
  '''
  Creates a series of temporal validation train and test sets
  Amount of train/test sets depends on length of split_thresholds
  
  Training and test set are delimited by the split_thresholds
  data_column indicates which column of dataframe (df) shall be used to compare with split_threshold value

  features contain features of data
  label_colum indicates which column is the output label
  test_window indicates length of test data
  gap_training_test indicates necessary distance between training and test data
  '''

  #Array to save train and test sets
  train_test_sets=[None] * len(split_thresholds)

  #For each threshold, create training and test sets
  for index, split_threshold in enumerate(split_thresholds):

    train_test_set={}
    train_test_set['id']=index
    train_test_set['split_threshold']=split_threshold

    #Columns of boolean values indicating if date_posted value is smaller/bigger than threshold
    
    #Train data is all data before threshold-gap
    train_filter = (df[data_column] < split_threshold-gap_training_test)

    #Test data is all thats test_window time after threshold
    test_filter = (df[data_column] >= split_threshold) & (df[data_column] < split_threshold+test_window)
    
    train_test_set['x_train'] = features[train_filter]
    train_test_set['y_train'] = df[label_column][train_filter]
    train_test_set['x_test'] = features[test_filter] 
    train_test_set['y_test'] = df[label_column][test_filter]
    
    train_test_sets[index]= train_test_set

  return train_test_sets


def get_models_and_parameters():

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'DT': DecisionTreeClassifier(random_state=0),

        'LR': LogisticRegression(penalty='l1', C=1),
        'SVM': LinearSVC(random_state=0, tol=1e-5, C=1, max_iter=10000.),

        'AB': AdaBoostClassifier(n_estimators=100),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
        'BA': BaggingClassifier(KNeighborsClassifier(),n_estimators=10)

        # 'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10), 
      }

    parameters_grid = { 
    'KNN': {'n_neighbors': [3,5,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,50,100],'min_samples_split': [2,5]},

    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'SVM': {'C' :[10**-2, 10**-1, 1 , 10, 10**2]}, #[10**-2, 10**-1, 1 , 10, 10**2]

    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'RF': {'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'BA': {'n_estimators': [10,100],'max_features': [1,10]}
    }
    

    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    # 'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    # 'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    # 'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    # 'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    # 'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01]},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BA': {'n_estimators': [10],'max_features': [1]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    }
    
    
    return models, test_grid


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

#This method expects ordered y_scores
def generate_binary_at_k(y_scores, k):    
    #Find the index position where the top k% finishes
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def metric_at_k(y_true, y_scores, k, metric):

    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    
    #generate binary y_scores
    binary_predictions_at_k = generate_binary_at_k(y_scores, k)


    # #classification_report returns different metrics for the prediction
    results = classification_report(y_true, binary_predictions_at_k, output_dict = True)
    
    if(metric=='precision'):
      metric = results['1']['precision']
    elif(metric=='recall'):
      metric = results['1']['recall']

    elif(metric=='f1'):
      metric = results['1']['f1-score']

    return metric

def generate_precision_recall_f1(y_test_sorted,y_pred_scores_sorted, thresholds):
  metrics = ['precision', 'recall', 'f1']

  output_array=[]

  for threshold in thresholds:
    for metric in metrics:
      metric_value = metric_at_k(y_test_sorted,y_pred_scores_sorted,threshold,metric)
      output_array.append(metric_value)
  return output_array

def iterate_over_models_and_training_test_sets(models_to_run, models, parameters_grid, train_test_sets):
  
  results_df =  pd.DataFrame(columns=(
    'model_name',
    'model',
    'parameters',
    'train_test_split_threshold',
    'p_at_1',
    'r_at_1',
    'f1_at_1',
    'p_at_2',
    'r_at_2',
    'f1_at_2',
    'p_at_5',
    'r_at_5',
    'f1_at_5',
    'p_at_10',
    'r_at_10',
    'f1_at_10',
    'p_at_20',
    'r_at_20',
    'f1_at_20',
    'p_at_30',
    'r_at_30',
    'f1_at_30',
    'p_at_50',
    'r_at_50',
    'f1_at_50',
    'auc-roc'))

  #For each training and test set
  for train_test_set in train_test_sets:

  # For each of our models
    for index,model in enumerate([models[x] for x in models_to_run]):
      print("Running "+str(models_to_run[index])+"...")
      
      #Get all possible parameters for the current model
      parameter_values = parameters_grid[models_to_run[index]]

      #For every combination of parameters
      for p in ParameterGrid(parameter_values):
        try:
            #Set parameters to the model. ** alows us to use keyword arguments
            model.set_params(**p)

            #Train model
            model.fit(train_test_set['x_train'], train_test_set['y_train'])
            
            #Predict
            y_pred_scores=0
            if(models_to_run[index] == 'SVM'):
              y_pred_scores = model.decision_function(train_test_set['x_test'])
            else:
              y_pred_scores = model.predict_proba(train_test_set['x_test'])[:,1]
            

            #Sort according to y_pred_scores, keeping map to their y_test values
            y_pred_scores_sorted, y_test_sorted = zip(*sorted(zip(y_pred_scores, train_test_set['y_test']), reverse=True))


            thresholds = [1,2,5,10,20,30,50]
            prec_rec_f1 = generate_precision_recall_f1(y_test_sorted,y_pred_scores_sorted, thresholds)

            roc_auc = roc_auc_score(train_test_set['y_test'], y_pred_scores)

            results_df.loc[len(results_df)] = [models_to_run[index],
                                               model,
                                               p,
                                               train_test_set['split_threshold']
                                               ]+prec_rec_f1+[roc_auc]
        except IndexError as e:
            print('Error:',e)

  return results_df