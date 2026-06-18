# 🏥 Health Status Machine Learning Pipeline (MLOps)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-orange.svg)](https://dvc.org/)
[![AWS S3](https://img.shields.io/badge/AWS-S3_Remote_Backend-yellow.svg)](https://aws.amazon.com/s3/)

## 🏗️ Architecture Overview
This repository contains a reproducible, configuration-driven machine learning pipeline for health status classification. It strictly separates source code from heavy data artifacts using **Data Version Control (DVC)** backed by an **AWS S3 Content-Addressable Storage (CAS)** system. 

The pipeline is built using modular Object-Oriented Programming (OOP) principles, allowing seamless tracking of data ingestion, preprocessing, feature engineering, and model training stages via a localized dependency graph (`dvc.yaml`).

## 🗂️ Project Structure

```text
health-status-ML/
├── .github/workflows/       # Automated CI/CD GitHub Actions (Pending)
├── data/                    # Local DVC data cache (Ignored by Git)
│   ├── raw/                 # Immutable source datasets
│   ├── processed/           # Cleaned intermediate arrays
│   └── final/               # Final scaled features for training
├── logs/                    # System execution and error logs
├── src/                     # Core Python OOP source modules
│   ├── data_ingestion.py    # Remote data fetch and validation wrappers
│   ├── data_preprocessing.py# Imputation and normalization logic
│   ├── feature_engg.py      # Feature selection and dimensionality reduction
│   └── model_training.py    # Dynamic Scikit-Learn model factory
├── dvc.yaml                 # DVC pipeline execution graph and dependency definitions
├── requirements.txt         # Locked dependency cryptographic versions
└── README.md                # System documentation