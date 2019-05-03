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

#for each value in 'values', we will find in which bin does it go
#according to the ranges specified in 'ranges'
#we return the bin index
def get_bin(values, ranges):
  ans =[]
  for value in values:
    found_bin=False
    for i in range(0,len(ranges)):
      if value < ranges[i]:
        ans.append(i)
        found_bin = True
        break

    if(not found_bin):
      ans.append(len(ranges))

  return ans

def get_bins_names(vector_indices,categories):
  ans =[]
  for i in vector_indices:
    ans.append(categories[i])
  return ans

def create_dummies(df, cols_to_transform):
  return pd.get_dummies(df, dummy_na=True, columns = cols_to_transform )


def create_discrete_feature(df, column, ranges, categories, new_column):
  print ("Creating discrete feature based on continuous variable...")
 
  indices = get_bin(df[column],ranges)

  df[new_column] = get_bins_names(indices,categories)

  print ("Done")
  return df

def extract_train_test_sets (df, test_start_time, train_end_time, date_column):

  train_set = df[df[date_column]<test_start_time]
  test_set = df[df[date_column]>train_end_time]

  return (train_set, test_set)

#5. Create temporal validation function in your pipeline that can create training and test sets over time. You can choose the length of these splits based on analyzing the data. For example, the test sets could be six months long and the training sets could be all the data before each test set.
def create_temp_validation_train_and_testing_sets(df, selected_features, outcome, start_time, end_time, prediction_window, date_column):

  start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
  end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

  train_start_time = start_time_date
  test_start_time = end_time_date - relativedelta(months=+prediction_window)
  train_end_time = test_start_time - relativedelta(days=+1)
  test_end_time = end_time_date
  
  train_and_test_set = extract_train_test_sets (df, test_start_time, train_end_time, date_column)

  train_set = train_and_test_set[0]
  test_set = train_and_test_set[1]

  # Now filter for selected columns
  x_train = train_set[selected_features]
  y_train = train_set[outcome]

  x_test = test_set[selected_features]
  y_test = test_set[outcome]

  return (x_train, x_test, y_train, y_test)

def get_train_and_testing_sets(df, selected_features, outcome, test_size):
  print ("Creating train and test sets...")

  x = df[selected_features]
  y = df[outcome]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

  print("Done")
  return x_train, x_test, y_train, y_test


def create_binary_feature(df, column, values_for_true_assignment, new_column):
  print ("Creating binary feature based on categorical variable...")




  col =[]

  for val in df[column]:
    if val in values_for_true_assignment:
      col.append(True)
    else:
      col.append(False)

  df[new_column] = col

  #Could be optimized using
  #df[new_column] = np.where(df[column] in values_for_true_assignment, True, False)
  
  print ("Done")
  return df

def get_train_and_testing_sets(df, selected_features, outcome, test_size):
  print ("Creating train and test sets...")

  x = df[selected_features]
  y = df[outcome]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

  print("Done")
  return x_train, x_test, y_train, y_test

def build_classifier(type_classifier, x_train, y_train, params=None):
  print ("Creating "+type_classifier+ " classifier...")


  if(type_classifier=='KNN'):
    # 'weights': ['uniform','distance']
    model = KNeighborsClassifier(n_neighbors=params[0], metric=params[1])
    

  elif(type_classifier=='LogisticRegression'):
    model = LogisticRegression(random_state=0, solver='liblinear', penalty=params[0], C=params[1])


  
  elif(type_classifier=='SVM'):
    model = LinearSVC(random_state=0, tol=1e-5, C=params[0])


# 'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]
  elif(type_classifier=='DecisionTree'):
    model = DecisionTreeClassifier(random_state=0)

  elif(type_classifier=='RandomForest'):
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
  elif(type_classifier=='Bagging'):
    model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
  elif(type_classifier=='Boosting'):
    model = AdaBoostClassifier(n_estimators=100)





  #Train model
  model.fit(x_train, y_train)

  print("Done")
  return model


def get_models_and_parameters():

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'DT': DecisionTreeClassifier(random_state=0),

        'LR': LogisticRegression(penalty='l1', C=1),
        'SVM': LinearSVC(random_state=0, tol=1e-5, C=1),

        'AB': AdaBoostClassifier(n_estimators=100),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
        'BA': BaggingClassifier(KNeighborsClassifier(),n_estimators=10)

        # 'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10), 
      }

    parameters_grid = { 
    'KNN': {'n_neighbors': [3,5,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,50,100],'min_samples_split': [2,5]},

    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'SVM': {'C' :[0.001,0.01,0.1,1,10]},

    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'RF': {'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'BA': {'n_estimators': [10,100],'max_features': [1,10]}
    }
    
    
    
    return models, parameters_grid


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

    return metric

def iterate_over_models(models_to_run, models, parameters_grid, x_train, x_test, y_train, y_test):
  
  results_df =  pd.DataFrame(columns=(
    'model_name',
    'model',
    'parameters',
    # 'train_set',
    # 'test_set',
    'p_at_1',
    'r_at_1',
    'p_at_2',
    'r_at_2',
    'p_at_5',
    'r_at_5',
    'p_at_10',
    'r_at_10',
    'p_at_20',
    'r_at_20',
    'p_at_30',
    'r_at_30',
    'p_at_50',
    'r_at_50',
    'auc-roc'))


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
            model.fit(x_train, y_train)
            
            #Predict
            y_pred_scores=0
            if(models_to_run[index] == 'SVM'):
              y_pred_scores = model.decision_function(x_test)
            else:
              y_pred_scores = model.predict_proba(x_test)[:,1]
            


            #Sort according to y_pred_scores, keeping map to their y_test values
            y_pred_scores_sorted, y_test_sorted = zip(*sorted(zip(y_pred_scores, y_test), reverse=True))



            results_df.loc[len(results_df)] = [models_to_run[index],
                                               model,
                                               p,

              metric_at_k(y_test_sorted,y_pred_scores_sorted,1.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,1.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,2.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,2.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,5.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,5.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,10.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,10.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,20.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,20.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,30.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,30.0,'recall'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,50.0,'precision'),
              metric_at_k(y_test_sorted,y_pred_scores_sorted,50.0,'recall'),
              roc_auc_score(y_test, y_pred_scores)
              ]
 
        except IndexError as e:
            print('Error:',e)

  return results_df