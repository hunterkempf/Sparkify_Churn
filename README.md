# Sparkify_Churn
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/1200px-Apache_Spark_logo.svg.png" align = "center" height="312" width="600" ></p>

Analysis of Churn in Sparkify Users. This Project uses PySpark and a Dataset provided by Udacity 

- [Overview of Files in Repo](#overview-of-files-in-repo)
- [Project Definition](#project-definition)
  * [Project Overview](#project-overview)
  * [Problem Statement](#problem-statement)
  * [Metrics](#metrics)
- [Libraries Used](#libraries-used) 
- [Analysis](#analysis)
  * [Data Exploration: Sample Data](#data-exploration-sample-data)
  * [Data Visualization: Sample Data](#data-visualization-sample-data)
  * [Data Exploration: Full Data](#data-exploration-full-data)
  * [Data Visualization: Full Data](#data-visualization-full-data)
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


## Overview of Files in Repo

 - Large-Spark-Capstone-ML.ipynb : Jupyter Notebook file with PySpark code analysis of 12 GB "Large" sized log data
 - Spark-Capstone-ML.ipynb : Jupyter Notebook File with PySpark code analysis of 242 MB "Medium" sized log data
 - medium-sparkify-event-data.json.zip : 242 MB "Medium" sized log data
 - Images/ : This folder contains screenshots of plots for the README.md Document



## Project Definition

### Project Overview

Customer Churn (cancelling a service) is an issue that many companies face. This problem is especially pervasive with streaming media companies like spotify or netflix that have low barriers to join or cancel service. I am going to predict customer churn using a large dataset of user activity. This project showcases what I have learned about working with relatively large datasets, building model features from user log data and optimizing machine learning models in PySpark. I am using data that was provided by Udacity's Data Science Nano Degree.

### Problem Statement

Sparkify is a music streaming company that is similar to Spotify or Pandora. They have users that stream music and every interaction with the platform is logged. This log data needs to be curated into a usable and feature rich dataset to build models that will predict users that are likely to churn. These predictions can be used by Sparkify to offer discounts or other incentives to users that are in danger of churning to try to retain them as customers.

### Metrics

Models will be evaluated on both the training data they were trained on and validation data that they were not trained on. Area Under the ROC Curve (AUC) will be the main metric used to chose between different models. AUC captures information from both True Positive Rate and False Positive Rate to provide a single score over a range of thresholds and is commonly used for classification models. 

## Libraries Used

General Python Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
```

PySpark SQL libraries
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, avg
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import FloatType
```

PySpark ML Libraries
```python
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.linalg import Vectors
```

## Analysis

I started my analysis with a smaller version of the dataset attached on the github as medium-sparkify-event-data.json.zip this allowed me to more quickly prototype my solution without the long run times of the 12 GB large dataset. In general I think that this is a good approach to analysis as long as the smaller dataset generally is representative of the larger dataset. 

### Data Exploration: Sample Data

I started my initial exploratory data analysis looking at the top page visits in the dataset.

![Top Page Visits](/Images/Top-Page-Visits.png)

From the summary the main page that is important right away is cancellation confirmation which I will use to determine if a user has churned. From the top pages visited it is also important to see the potential information we can aggregate and understand about a user's behavior such as the number of songs they listened to, the number of positive/negative ratings they have given, the humber of friends they have on the platform and if they are experiencing errors.

![Churn Raw](/Images/Churn-raw.png)

As you can see out of our total number of 448 users 349 are subscribers and 99 are no longer subscribers and have churned. This is a bit unbalanced but not as bad as many fraud datasets. What is not ideal for a machine learning algorithm is that despite starting with what seemed to be a resonably large dataset of 543,705 rows of logs there are effectively only 448 data points in our model.


### Data Visualization: Sample Data

![Churn By Gender](/Images/Churn-by-gender.png)

![CountThumbsUp EDA](/Images/CountThumbsUp-EDA.png)

### Data Exploration: Full Data

For the Full Dataset I repeated much of the EDA I did for the smaller dataset but wanted to see how the analysis would differ or remain the same so much of the views will be the same. 

![Top Page Visits Full](/Images/Top-Page-Visits-full.png)

From the summary the main page that is important right away is cancellation confirmation which I will use to determine if a user has churned. From the top pages visited it is also important to see the potential information we can aggregate and understand about a user's behavior such as the number of songs they listened to, the number of positive/negative ratings they have given, the humber of friends they have on the platform and if they are experiencing errors.

![Churn Raw Full](/Images/Churn-raw-full.png)

As you can see out of our total number of 22,278 users 17,275 are subscribers and 5,003 are no longer subscribers and have churned. This is roughly the same distribution as the sample dataset which had 22.1% of the users churn and the full dataset has 22.5% churn. We still started out with a much larger dataset of 26,259,199 rows of logs than the final model ready dataset of 22,278 data points but 22k datapoints should be enough to really train some of the more advanced models which makes the larger dataset much more attractive to use for our final model.

### Data Visualization: Full Data

![Churn By Gender Full](/Images/Churn-by-gender-full.png)

![CountThumbsUp EDA Full](/Images/CountThumbsUp-EDA-full.png)

## Methodology

### Data Preprocessing

The dataset of log records needs a lot of preprocessing to create a dataset that is model ready for a machine learning algorithm. 

The first step that needs to be done is to create a truth set. This means for our case that we need to look at the cases where a user has churned. If a user visits the page Cancellation Confirmation it can be inferred that they have cancelled their service. This means that for our labels we can look at every user and determine if they visited that page or not and that will be the basis for our dataset. 

Because our dataset is aggregated at the userId level due to our label variable, all of our model features should be aggregated at the userId level as well. This means most of our data points will have to be counts, averages or sums. In my case I chose to build 14 aggregate features. 

### Implementation

Because of the size of the full dataset (12 GB), Spark and more specifically PySpark was chosen for the implementation of the Preprocessing because of how well it performs on large datasets. It can be run on single machines (which may run slowly) or on clusters on premise or in the public cloud (AWS, AZURE, Google, IBM) which allows for the same code to be run in any different environment. 

Here is an example of an aggregated feature using PySpark:

First we create an aggregated dataset based on our criteria
```python
df = df.withColumn('ThumbsUp', (when(col('page')== 'Thumbs Up',1)\
                                                            .otherwise(0)))
user_thumbsUp_df = df.groupby('userId')\
                       .agg(_sum('ThumbsUp')\
                       .alias('countThumbsUp'))

```
Then we join the now aggregated data back to the labeled truth set
```python
# Join data back to our user dataframe
user_labeled_df = user_labeled_df.join(user_thumbsUp_df, 'userId')
user_labeled_df.show(5)
```

### Refinement

The Features that I have built will be refined and weighted by the PySpark ML library models that I use to make the churn predictions. If a feature is not predictive for a certain type of model it will either have a very low weight or a weight of zero. 

Those models can also be refined and hyperparameter tuned to change how they interact with the variables.

## Results

### Model Evaluation and Validation

#### Logistic Regression

Basic Model

![Logistic Regression Train AUC](/Images/Logistic-Regression-Training-AUC.png)

![Logistic Regression Test AUC](/Images/Logistic-Regression-Test-AUC.png)

This model is basic but does not appear to be overfit because the AUC for training and test datasets appear to be roughly the same. The nice thing about logistic regression is that it is a simple model that can be resonably interpretable. In order to optimize the model performance a parameter grid search can be used to use some elastic net regularization as well as if an intercept should be fit or not. That should most likely make a more generalizable model that performs well.

Grid Search Hyper Parameter Tuned Model

![Logistic Regression Train AUC tuned](/Images/Logistic-Regression-Training-AUC-tuned.png)

![Logistic Regression Test AUC tuned](/Images/Logistic-Regression-Test-AUC-tuned.png)

Hyper-parameter tuning results in roughly the same outcome as the model defaults. In this case we could expand our grid search or try a different model.

#### Desicion Tree

#### Gradient Boosted Tree (GBTree)

Basic Model

![GBTree Train AUC](/Images/GBTree-Training-AUC.png)

![GBTree Test AUC](/Images/GBTree-Test-AUC.png)

This model improves the best AUC we were able to achieve on the training and test sets that we were able to get with Logistic Regression or Decision Trees! In order to combat the small overfitting we see with the drop between training and test AUC a parameter grid search can be used to try to optimize hyperparameter tuning similar to what we have done for the Logistic Regression above. Usually this will result in a more generalizable model.

Grid Search Hyper Parameter Tuned Model

![GBTree Train AUC tuned](/Images/GBTree-Training-AUC-tuned.png)

![GBTree Test AUC tuned](/Images/GBTree-Test-AUC-tuned.png)

This model with parameter tuning and a validation split performs a little worse than the GBTree model with the default parameters in training but you can see that there is basically no drop off in the test set. I think that this model would be better to use and would generally perform more reliably than the original GBTree model we had.

### Justification

For this analysis we tried 3 different models and compared them on the same metric of AUC. Ultimately GBTree proved to be the best algorithm to use for our model. Using a Grid Search and Train/Validation Splits during the training process we were able to find optimal parameters that did not show signs of overfitting. 

## Conclusion

### Reflection

In Conclusion, I took a large dataset of user behavior logs, created a model ready dataset using PySpark and used that dataset to predict if a user is likely to churn. I really enjoyed getting to work with a large dataset and trying out jupyter notebooks hosted on a cloud provider. One aspect I found difficult was the time it takes for larger datasets to run and the limited selection of models in PySpark compared to Scikit Learn.

### Improvement

Future Improvements to this model can be from the following areas:

- Spark Structured Streaming: For a real usecase it would be important to gain the understanding that a user is likely to churn right away thus it would be a good idea to use Spark's Structured Streaming API to get model scores back in near real time.

- Time Based Truth Set: Since the data is about user churn it would be better if I created a truth set on user behavior for a certain time ie did a user churn in the next week from the features. This would require significantly more work on the data engineering side but could allow for a more realistic dataset of the propensity to churn problem.

- Date Based Features: The features I created were all relatively simple so including some features based on the number of days or months that a user has been on the service would be useful ie logins per month or songs per month. 

- Trending Features: What a user has done in the last session, last day, last week etc may have predictive power that would allow for more insights to be found. Adding these trending features could provide an improvement to predictions we are making.

- Unbalanced Data Corrections: We have a relatively unbalanced dataset ~22% of our users have churned. We could use oversampling, undersampling or model weights to try to correct for this.

- XGBoost: I really like using XGBoost due to its predictive power. PySpark doesnt have an implementation of XGBoost yet so if I imported the dataset to pandas and ran XGBoost on the dataframe I believe I would gain some benefits of using a better algorithm.

- H2O Sparkling Water: H2O created an opensource connector for using an H2O trained model in spark. Given H2O's prebuilt functionality it would improve both the model selection I have to chose from and the Grid Search over hyper-parameters that I would have to do.

