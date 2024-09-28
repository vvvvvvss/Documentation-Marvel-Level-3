# Documentation-Marvel-Level-3
## Task 1 - Decision Tree based ID3 Algorithm
To predict the quality of wine, I used Decision Tree using the ID3 algorithm.
I built the ID3 algorithm from scratch which involved calculating entropy, information gain, and weighted average entropy to identify the optimal feature for each split.
Features like acidity and alcohol content, were important predictors of wine quality.  
[Code](https://github.com/vvvvvvss/Decision-Tree-based-ID3-Algorithm)

## Task 2: Naive Bayesian Classifier
Built a  Naive Bayesian Classifier that works on BBC's data and categorizers texts into entertainment, tech, business, sport, etc. 
The categorical data is converted into numerical data such that it can be interpreted by the machine. 
This is done by analyzing repeated words in each category. The categorical data is converted into numerical data.  
[Code](https://github.com/vvvvvvss/Naive-bayes/tree/main)

## Task 3 - Ensemble techniques  
Applied Ensemble Techniques to the titanic dataset. To this dataset, I incooperated all three ensemble techniques.
Ensemble learning refers to algorithms that combine the predictions from two or more models. 
Combining models like Random Forest, Decision Trees, and Gradient Boosting increased the accuracy of the model.  
[Code](https://github.com/vvvvvvss/EnsembleTechniques/tree/main)

## Task 4 - Random Forest, GBM and Xgboost
### 1. Random Forest
Used a random foreset classifier to predict if a patient is with heart disease. Random Forest Classifiers are a collection of individual decision trees. 
More is uncorrelation between the Decision trees , more is the accuracy of the Random Forest classifier.  
[Code](https://github.com/vvvvvvss/RandomForestClassifier)

### 2. GBM
Used Gradient Boosting Classifier to predict if a patient is with breast cancer or not.  In GBM, the week learning models combine with the stronger learning models. 
Boosting is one kind of ensemble Learning method which trains the model sequentially and each new model tries to correct the previous model.  
[Code](https://github.com/vvvvvvss/GradientBoostingClassifier)

### 3. XGBoost
XGBoost is also an Ensemble learning method, that stands for Extreme Gradient Boosting. Here, I've used XGBoost to predict the if a person would return the loan he/she has taken from the bank.  
[Code](https://github.com/vvvvvvss/XGBoost)  

## Task 5 - Hyperparameter Tuning
Used Hyperparameter tuning to increase the accurcy from 81% to 92% of Student performance dataset. 
I first create a parameter grid that tries different values of each parameters such as max_depth, min_samples_split, min_samples_leaf, criterion and find the fest values for each parameter.  
[Code](https://github.com/vvvvvvss/Hyperparameters/blob/main/StudentPerformance.ipynb])  
### Before Hyperparameter Tuning
![image](https://github.com/user-attachments/assets/dcfae1e1-3259-4e67-8410-64cbd4318173)
### After Hyperparameter Tuning
![image](https://github.com/user-attachments/assets/d11ebeb8-5271-458b-a7b4-53218ddf1a99)

## Task 6 : Image Classification using KMeans Clustering
K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into a pre-defined number of clusters.
This first step for this task was to find the 'k' value and initialize the centroids.
Next, by converting the images into numerical data and applying clustering, I classified image segments.  
[Code](https://github.com/vvvvvvss/KMeansClustering/tree/main)  
![image](https://github.com/user-attachments/assets/2f05fcd5-7dac-4d3b-8a67-4a88efe8d731)  


## Task 7: Anomaly Detection
I learned about anomaly detection techniques and implemented them on the load_linnerud dataset(toy dataset), which contains physiological and exercise data.
I applied the Local Outlier Factor (LOF) algorithm  to identify outliers in the dataset. 
In this datset the outliers would be unusual physiological responses or extreme exercise measurements.  
[Code](https://github.com/vvvvvvss/AnomalyDetection/tree/main)  
  
![image](https://github.com/user-attachments/assets/d7f580d9-a52a-4979-b153-4b6f247add64)  

