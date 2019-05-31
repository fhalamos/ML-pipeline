# ML-pipeline

This repository presents a Machine Learning Pipeline to solve ML problems. It also presents a use case for predicting if projects from donorschoose will not be funded in 60 days.

## How to use?
To use the pipeline, import pipeline_v2.py in your use case and use the following methods:

* read_csv() to load files
* explore_data() to get basic stats and data exploration
* fill_na_columns_with_mean for data cleaning
* create_dummies() for creating dummy variables from categorical data
* create_discrete_features() for creating discrete features from continuous data
* create_temp_validation_train_and_testing_sets() to generate a series of training and test sets for temporal validation
* get_models_and_parameters() to get varios ML models and their possible parameters
* iterate_over_models_and_training_test_sets() to run varios ML models over different train/test sets and evaluate them

## Files in this repo

* pipeline_v2.py: python ML pipeline
* predict_donorschoose_60_days: implementation of pipeline_v2 for the donorschoose problem
* projects_2012_2013.csv: dataset
* Plots folder: contains all precision-recall curves generated in this implementation
* Report.pdf contains a report of results for this implementation

A minimalistic pipeline and its use case can be found in the minimalist_pipeline folder

For the minimalistic problem (HW2), under minimalistic_pipeline folder
* pipeline.py: minimalistic ML pipeline
* predicting_financial_credit.ipynb: implementation of pipeline.py
* credit-data.csv:	dataset
* Dictionary.xls: dataset dictionary for hw2
