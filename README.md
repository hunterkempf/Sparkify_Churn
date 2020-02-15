# Sparkify_Churn
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/1200px-Apache_Spark_logo.svg.png" align = "center" height="312" width="600" ></p>

Analysis of Churn in Sparkify Users. This Project uses PySpark and a Dataset provided by Udacity 

- [Project Definition](#project-definition)
  * [Project Overview](#project-overview)
  * [Problem Statement](#problem-statement)
  * [Metrics](#metrics)
- [Analysis](#analysis)
  * [Data Exploration](#data-exploration)
  * [Data Visualization](#data-visualization)
- [Methodology](#methodology)
  * [Data Preprocessing](#data-preprocessing)
  * [Implementation](#implementation)
  * [Refinement](#refinement)
- [Results](#results)
  * [Model Evaluation and Validation](#model-evaluation-and-validation)
  * [Justification](#justification)
- [Conclusion](#conclusion)
  * [Reflection](#reflection)
  * [Improvement](#improvement)  

## Project Definition

### Project Overview

Customer Churn (cancelling a service) is an issue that many companies face. This problem is especially pervasive with streaming media companies like spotify or netflix that have low barriers to join or cancel service. I am going to predict customer churn using a large dataset of user activity. This project showcases what I have learned about working with relatively large datasets, building model features from user log data and optimizing machine learning models in PySpark. I am using data that was provided by Udacity's Data Science Nano Degree.

### Problem Statement

Sparkify is a music streaming company that is similar to Spotify or Pandora. They have users that stream music and every interaction with the platform is logged. This log data needs to be curated into a usable and feature rich dataset to build models that will predict users that are likely to churn. These predictions can be used by Sparkify to offer discounts or other incentives to users that are in danger of churning to try to retain them as customers.

### Metrics

Models will be evaluated on both the training data they were trained on and validation data that they were not trained on. Area Under the ROC Curve (AUC) will be the main metric used to chose between different models. AUC captures information from both True Positive Rate and False Positive Rate to provide a single score over a range of thresholds and is commonly used for classification models. 

## Analysis

### Data Exploration



### Data Visualization

## Methodology

### Data Preprocessing

### Implementation

### Refinement

## Results

### Model Evaluation and Validation

### Justification

## Conclusion

### Reflection

### Improvement
