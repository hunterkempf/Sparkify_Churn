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

I started my initial exploratory data analysis looking at the top page visits in the dataset.

![Top Page Visits](/Images/Top-Page-Visits.png)

From the summary the main page that is important right away is cancellation confirmation which I will use to determine if a user has churned. From the top pages visited it is also important to see the potential information we can aggregate and understand about a user's behavior such as the number of songs they listened to, the number of positive/negative ratings they have given, the humber of friends they have on the platform and if they are experiencing errors.

![Churn Raw](/Images/Churn-raw.png)

As you can see out of our total number of 448 users 349 are subscribers and 99 are no longer subscribers and have churned. This is a bit unbalanced but not as bad as many fraud datasets. What is not ideal for a machine learning algorithm is that despite starting with what seemed to be a resonably large dataset of 543705 rows of logs there are effectively only 448 data points in our model.



### Data Visualization

![Churn By Gender](/Images/Churn-by-gender.png)


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
