# ML-pipeline

Basic Machine Learning Pipeline, and use case for predicting financial credits

How to use?

import pipeline_v2.py in your use case and use the following methods:

* read_csv() to load files
* explore_data() to get basic stats and data exploration
* pre_process_data() for data cleaning
* create_dummies() for creating dummy variables from categoricals
* get_models_and_parameters():
* iterate_over_models() to run varios ML models and evaluate them

pipeline.py is only the simple/old version of pipeline_v2.py used in HW2

Files in this repo
* pipeline_v2.py: python ML pipeline
* predict_donorschoose_60_days: implementation of pipeline_v2 (HW3)
* predicting_financial_credit.ipynb: implementation of pipeline.py (HW2)
* credit-data.csv:	dataset for hw2
* projects_2012_2013.csv: dataset for hw3
* Dictionary.xls: dataset dictionary for hw2
* Homework 2 – Machine Learning Pipeline.pdf: homework instructions
