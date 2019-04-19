# ML-pipeline

Basic Machine Learning Pipeline, and use case for predicting financial credits

How to use?

import pipeline.py in your use case and use the following methods:

* read_csv() to load files
* explore_data() to get basic stats and data exploration
* pre_process_data() for data cleaning
* create_discrete_feature() to discretize a continuous variable and create a new discrete one
* create_binary_feature() to create a binary variable based on discrete variable
* get_train_and_testing_sets()
* build_classifier(), choosing your prefered model (only logistic regression at the moment)
* evaluate_classifier()

Files in this repo
* pipeline.py: python ML pipeline
* predicting_financial_credit.ipynb: implementation of the ML pipeline in predicting financial credit
* credit-data.csv:	credits dataset
* Dictionary.xls: dataset dictionary
* Homework 2 – Machine Learning Pipeline.pdf: homework instructions
