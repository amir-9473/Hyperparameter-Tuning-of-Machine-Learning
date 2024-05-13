![image](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/assets/81913183/c23db1bc-250b-4148-827f-e7791d5f080a)
# Hyperparameter Tuning of Machine Learning Algorithm
[![GitHub license](https://img.shields.io/github/license/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg)](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg)](https://GitHub.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/graphs/contributors/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg)](https://GitHub.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/pulls/) 

[![GitHub watchers](https://img.shields.io/github/watchers/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg?style=social&label=Watch)](https://GitHub.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learnings/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg?style=social&label=Fork)](https://GitHub.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/network/)
[![GitHub stars](https://img.shields.io/github/stars/amir-9473/Hyperparameter-Tuning-of-Machine-Learning.svg?style=social&label=Star)](https://GitHub.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/stargazers/)


Hyperparameter Optimization for Pima Indians Diabetes Machine Learning Classification using Optuna in Python.


![](https://ch-api.healthhub.sg/api/public/content/d9607c272f414aa280163d4081417dd0?v=548317ad&t=livehealthyheaderimage)


## Quick Access

Quick access to the data is provided, including:

- The raw dataset.
- The cleaned data after preprocessing.
- The trained model saved as a pickle (.pkl) file.
- Jupyter notebooks containing the code and exploration of the project.

| Dataset (raw) | Cleaned Data   | Trained Models      | Notebooks        |
|---------------|----------------|-------------|------------------|
| [Pima (csv)](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/blob/main/data/raw/diabetes.csv)     | [data_cln1(csv)](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/blob/main/data/processed/data_cln1.csv) | [models (pkl)](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/tree/main/models) | [notebooks(ipynb)](https://github.com/amir-9473/Hyperparameter-Tuning-of-Machine-Learning/tree/main/notebooks) |


## Problem definition

### What is diabetes?

Diabetes, a chronic health condition, affects the body's ability to process food into energy. The process involves breaking down food into sugar and releasing it into the bloodstream, regulated by insulin. In diabetes, the body either lacks sufficient insulin production or becomes resistant to its effects, leading to elevated blood sugar levels and potential health complications. The three main types of diabetes are type 1, type 2, and gestational diabetes, each with distinct characteristics.

* **Type 1 diabetes results from an autoimmune reaction that impairs insulin production, usually diagnosed in children and young adults.**

* **Type 2 diabetes involves insulin resistance and is commonly diagnosed in adults, but increasingly in younger populations.**

Gestational diabetes occurs during pregnancy and may elevate the risk of health issues for both the mother and the baby. While there is no cure for diabetes, lifestyle changes such as weight management, healthy eating, and physical activity can greatly improve its management and prevention.
## Dataset (Pima Indian Diabetes)

**What is Pima Indian Diabetes?**
* **Pima Indian diabetes**, prevalent among the Pima Native American population in Arizona, is characterized by a remarkably high incidence of type 2 diabetes. Its study offers unique insights into genetic and environmental factors contributing to the disease, with broader implications for diabetes research and healthcare. Understanding this condition can aid in developing effective healthcare practices and prevention strategies, making it a critical focus of medical investigation.

* **Pima Indians Diabetes Dataset** aims to predict diabetes onset using diagnostic measures. The dataset, sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, focuses on female patients aged 21 or older with Pima Indian heritage. Its objective is to diagnostically predict diabetes presence based on specific diagnostic measurements, carefully selected from a larger database.

Attribute Information:

1. Number of times pregnant

2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test

3. Diastolic blood pressure (mm Hg)

4. Triceps skin fold thickness (mm)

5. 2-Hour serum insulin (mu U/ml)

6. BMI (weight in kg/(height in m)^2)

7. Diabetes pedigree function

8. Age (years)

9. Class variable (0 means non-diabetic or 1 means diabetic)


for more information and to download this dataset, please refer to the following link:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
## What is the hyperparameter tuning?

* **Parameters** in a machine learning model are algorithm-produced variables that create predictions based on the training data.

* **Hyper-parameters**, on the other hand, are user-defined variables while building the model, controlling the learning process.

Hyperparameter optimization aims to find the best solution among all possibilities, optimizing the target value. Packages like GridSearchCV, RandomizedSearchCV, and Optuna assist in identifying the optimal hyperparameter combinations for our models. These techniques are referred to as hyperparameter tuning or optimization.

If we don't specify parameter values, default values are used, raising the question of how suitable they are for the dataset, which is crucial to consider during model building.

**Basic Techniques:**
* RandomizedSearchCV
* GridSearchCV

**Advanced Techniques:**
* Bayesian Optimization
* TPOT Classifier(Genetic Algorithm)
* Optuna
* etc

## Why Optuna?

Optuna is a versatile hyperparameter tuning library, designed to work with any machine learning or deep learning framework. It offers three key features for optimal hyperparameter optimization:

1. Eager search spaces: Automates the search for optimal hyperparameters.
2. State-of-the-art algorithms: Efficiently explores large parameter spaces and prunes unproductive trials for faster results.
3. Easy parallelization: Enables hyperparameter search across multiple threads or processes without code modification.

In this guide, we'll begin by using Optuna with sklearn, a user-friendly machine learning framework.
## Model Performance
Four basic Machine Learning models were employed in this project, and their hyperparameters were optimized using the Optuna tool. The final accuracy outputs of the models are as follows:

| Model    | KNN | SVM | DCT | LR |
|----------|-----|-----|-----|----|
| Accuracy | 92  | 89  | 95  | 83 |

![](https://i.imgur.com/ddEx6eD.jpeg)
