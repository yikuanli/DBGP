# Deep Bayesian Gaussian Processes for Uncertainty Estimation in Electronic Health Records

# Introduction
This project contains several model structures for uncertainty estimation, including BEHRT, BEHRT with Bayesian embedding, BEHRT with Bayesian embedding and classifier, BEHRT with Bayesian classifier, KISS-GP, whitened-GP, and DBGP, all GP based models are also based on BEHRT, which can be found in arXiv:1907.09538. 

# Description
common: file for spark setup, common functions in general (file store and load)
CPRD: Basic functions to process CPRD tables
dataLoader: DataLoader for prediction tasks
evaluation: some evaluation metrics for generalisation measurement, uncertainty measurement.
model: folder contains several model architectures and corresponding optimisers for trianing
preprocessing: include code for cohort selection and some data pre-processing, and a data generator for generating data for prediction task
task: include scripts for model training and model performance analysis


