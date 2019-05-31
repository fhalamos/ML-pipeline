import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as accuracy
import seaborn as sns


pd.options.mode.chained_assignment = None  # default='warn'


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


def create_discrete_feature(df, column, ranges, categories, new_column):
  print ("Creating discrete feature based on continuous variable...")
 
  indices = get_bin(df[column],ranges)

  df[new_column] = get_bins_names(indices,categories)

  print ("Done")
  return df



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


def get_train_and_testing_sets(df, selected_features, outcome):
  print ("Creating train and test sets...")

  x = df[selected_features]
  y = df[outcome]

  test_size = 0.3
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

  print("Done")
  return x_train, x_test, y_train, y_test


def build_classifier(type_classifier, x_train, y_train):
  print ("Creating "+type_classifier+ " classifier...")


  if(type_classifier=='LogisticRegression'):

    #Select logistic model
    logisticRegr = LogisticRegression()

    #Train model
    logisticRegr.fit(x_train, y_train)

    print("Done")
    return logisticRegr


def evaluate_classifier(model,x_test,y_test,threshold):
  #Using accuracy: correct predictions / total number of predictions
  print ("Evaluating classifier...")
  pred_scores = model.predict_proba(x_test)[:,1]

  pred_label = [1 if x>threshold else 0 for x in pred_scores]
 
  correct_pred = [1 if pred_label[i]==y_test.values[i] else 0 for i in range(0,len(pred_label))]

  accuracy = sum(correct_pred)/len(correct_pred)
  #accuracy = model.score(x_test, y_test)
  
  print("(Threshold: {}), Predicted correct {} out of {}, the accuracy is {:.3f}".format(
      threshold, sum(correct_pred), len(y_test.values), accuracy))

  print("Done")