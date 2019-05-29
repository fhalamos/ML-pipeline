# Based on https://github.com/rayidghani/magicloops

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import *
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_curve

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import textwrap

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

def create_temp_validation_train_and_testing_sets(df, features, data_column, label_column, split_thresholds, test_window, gap_window):
  '''
  Creates a series of temporal validation train and test sets
  Amount of train/test sets depends on length of split_thresholds array
  
  Training and test set are delimited by the split_thresholds
  data_column indicates which column of dataframe (df) shall be used to compare with split_threshold value

  features contain features of data
  label_colum indicates which column is the output label
  test_window indicates length of test data
  gap_window indicates necessary time we need for train and test data to look at outcome (do not include data whose date_posted is in gap time hence)
  '''

  #Array to save train and test sets
  train_test_sets=[None] * len(split_thresholds)

  #For each threshold, create training and test sets
  for index, split_threshold in enumerate(split_thresholds):

    train_test_set={}
    train_test_set['test_set_start_date']=split_threshold

    #Columns of boolean values indicating if date_posted value is smaller/bigger than threshold
    
    #Train data is all data before threshold-gap
    train_filter = (df[data_column] < split_threshold-gap_window)

    #Test data is all data thats after training data(after split_threshold), but only consider a length of test_window time, - necessary gap to see all outcomes.
    test_filter = (df[data_column] >= split_threshold) & (df[data_column] < split_threshold+test_window-gap_window)
    
    train_test_set['x_train'] = features[train_filter]
    train_test_set['y_train'] = df[label_column][train_filter]
    train_test_set['x_test'] = features[test_filter] 
    train_test_set['y_test'] = df[label_column][test_filter]
    
    train_test_sets[index]= train_test_set

  return train_test_sets


def get_models_and_parameters():

    models = {

      'DT': DecisionTreeClassifier(random_state=0),
      'LR': LogisticRegression(penalty='l1', C=1),
      'RF': RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),

      'BA': BaggingClassifier(KNeighborsClassifier(),n_estimators=10),
      'AB': AdaBoostClassifier(n_estimators=100),
      'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
      'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
      
      'SVM': LinearSVC(random_state=0, tol=1e-5, C=1, max_iter=10000),
      'KNN': KNeighborsClassifier(n_neighbors=3),
      'NB': GaussianNB()
    }

    parameters_grid = { 
  
      'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,50,100],'min_samples_split': [2,5]},
      'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
      'RF': {'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},

      'BA': {'n_estimators': [10,100],'max_features': [1,10]},
      'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
      'GB': {'n_estimators': [100, 10000], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
      'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},

      'SVM': {'C' :[10**-2, 10**-1, 1 , 10, 10**2]}, 
      'KNN': {'n_neighbors': [3,5,10,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree']},
      'NB' : {}
    }
    

    test_grid = { 

      'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
      'LR': { 'penalty': ['l1'], 'C': [0.01]},
      'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},

      'BA': {'n_estimators': [10],'max_features': [1]},
      'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
      'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
      'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},

      'SVM' :{'C' :[0.01]},
      'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
      'NB' : {}

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

def plot_precision_recall_n(y_true, y_score, model, parameter_values, test_set_start_date, output_type='save'):

    '''
    Plot precision recall curves
    -y_true contains true values
    -y_score contains predictions
    -model is the model being run
    -parameter_values contains parameters used in this model: we will use this for the plot name
    -output_type: either saving plot or displaying
    '''

    #Compute precision-recall pairs for different probability thresholds
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score) 

    #The last precision and recall values are 1. and 0 in precision_recall_curve method, now removing them 
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]


    #We transform the pr_thresholds (which is an array with scores thresholds, to an array of percentage thresholds)
    pct_above_per_thresh = []
    number_scored = len(y_score)    
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    #Clear any existing figure
    plt.clf()

    #Create a figure and access to its axis
    fig, ax1 = plt.subplots()

    #Create blue line for precision curve
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')

    #Create a duplicate axis, and use it to plot recall curve
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
 
    #Limit axis borders
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,1])
    ax2.set_ylim([0,1])
    
    #Set name of plot 
    model_name = str(model).split('(')[0]
    chosen_params = str(parameter_values)
    plot_name = model_name+'-'+chosen_params+'-test_set_start_date:'+test_set_start_date


    #Set title and position in plot
    title = ax1.set_title(textwrap.fill(plot_name, 70))
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)    

    #Save or show plot
    if (output_type == 'save'):
        plt.savefig('Plots/'+str(plot_name)+'.png')
    elif (output_type == 'show'):
        plt.show()
    plt.close()


def iterate_over_models_and_training_test_sets(models_to_run, models, parameters_grid, train_test_sets):
  
  results_df =  pd.DataFrame(columns=(
    'model_name',
    'model',
    'parameters',
    'test_set_start_date',
    'baseline',
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

      #Get all possible parameters for the current model
      parameter_values = parameters_grid[models_to_run[index]]

      #print("Running "+str(models_to_run[index])+" with params: "+str(parameter_values) +" on train/test set "+str(train_test_set['test_set_start_date']))
      

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


            thresholds_for_metrics = [1,2,5,10,20,30,50]

            baseline = metric_at_k(y_test_sorted,y_pred_scores_sorted,100,'precision')

            prec_rec_f1 = generate_precision_recall_f1(y_test_sorted,y_pred_scores_sorted, thresholds_for_metrics)

            roc_auc = roc_auc_score(train_test_set['y_test'], y_pred_scores)

            test_set_identifier = str(train_test_set['test_set_start_date']).split(' ')[0]

            results_df.loc[len(results_df)] = [models_to_run[index],
                                               model,
                                               p,
                                               test_set_identifier,
                                               baseline
                                               ]+prec_rec_f1+[roc_auc]
            
            plot_precision_recall_n(train_test_set['y_test'],y_pred_scores,model,p,str(train_test_set['test_set_start_date']),'save')


        except IndexError as e:
            print('Error:',e)

  return results_df