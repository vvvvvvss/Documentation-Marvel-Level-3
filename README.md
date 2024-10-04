# Documentation-Marvel-Level-3
## Task 1 - Decision Tree based ID3 Algorithm
To predict the quality of wine, I used Decision Tree using the ID3 algorithm.
I built the ID3 algorithm from scratch which involved calculating entropy, information gain, and weighted average entropy to identify the optimal feature for each split.
Features like acidity and alcohol content, were important predictors of wine quality.  
[Code](https://github.com/vvvvvvss/Decision-Tree-based-ID3-Algorithm)  
![image](https://github.com/user-attachments/assets/3e2eea37-d419-411f-9e28-2c0d640a677a)


## Task 2: Naive Bayesian Classifier
Built a  Naive Bayesian Classifier that works on BBC's data and categorizers texts into entertainment, tech, business, sport, etc. 
The categorical data is converted into numerical data such that it can be interpreted by the machine. 
This is done by analyzing repeated words in each category. The categorical data is converted into numerical data.  
[Code](https://github.com/vvvvvvss/Naive-bayes/tree/main)  
![image](https://github.com/user-attachments/assets/4fafc17b-0b81-43c7-ac78-caf6d1243e25)


## Task 3 - Ensemble techniques  
Applied Ensemble Techniques to the titanic dataset. To this dataset, I incooperated all three ensemble techniques.
Ensemble learning refers to algorithms that combine the predictions from two or more models. 
Combining models like Random Forest, Decision Trees, and Gradient Boosting increased the accuracy of the model.  
[Code](https://github.com/vvvvvvss/EnsembleTechniques/tree/main)   
![image](https://github.com/user-attachments/assets/fc3131b6-45db-4744-ae86-f2ad938ad612)


## Task 4 - Random Forest, GBM and Xgboost
### 1. Random Forest
Used a random foreset classifier to predict if a patient is with heart disease. Random Forest Classifiers are a collection of individual decision trees. 
More is uncorrelation between the Decision trees , more is the accuracy of the Random Forest classifier.  
[Code](https://github.com/vvvvvvss/RandomForestClassifier)  
![image](https://github.com/user-attachments/assets/67edba77-f3a0-4242-9a5c-9dcbf51a5923)


### 2. GBM
Used Gradient Boosting Classifier to predict if a patient is with breast cancer or not.  In GBM, the week learning models combine with the stronger learning models. 
Boosting is one kind of ensemble Learning method which trains the model sequentially and each new model tries to correct the previous model.  
[Code](https://github.com/vvvvvvss/GradientBoostingClassifier)  
![image](https://github.com/user-attachments/assets/1e762e7e-b084-4e89-8782-d87d1126671b)


### 3. XGBoost
XGBoost is also an Ensemble learning method, that stands for Extreme Gradient Boosting. Here, I've used XGBoost to predict the if a person would return the loan he/she has taken from the bank.  
[Code](https://github.com/vvvvvvss/XGBoost)   
![image](https://github.com/user-attachments/assets/cbd7a6a6-e8af-4626-bc76-f08c4b501a72)


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

## Task 8: Generative AI Task Using GAN
In GAN two models are trained simultaneously. These two models are Generator and discriminator. 
The generator is the artist that learns to create images that look real, while a discriminator is the art critic that learns to tell real images apart from fakes.
During training, the generator progressively becomes better at creating images that look real, while the discriminator becomes better at telling them apart. 
The process reaches equilibrium when the discriminator can no longer distinguish real images from fakes.
[Code](https://github.com/vvvvvvss/GAN/tree/main)  
![image](https://github.com/user-attachments/assets/f20edb03-7cc2-425b-bef4-eb845f8c8dc5)

## Task 9: PDF Query Using LangChain
LangChain is a framework that allows developers to create agents capable of reasoning about issues and breaking them down into smaller sub-tasks. I used libraries such as 
HuggingFaceEmbeddings which uses sentence-transformers to generate embeddings for the text chunks, and RetrievalQA Chain which combines retrieval with the LLM to provide answers to your questions.
[Code](https://github.com/vvvvvvss/PdfQuery)  
![image](https://github.com/user-attachments/assets/a0313cee-e4c4-49f6-b8e3-879295cde33e)

## Task 10: Table Analysis Using PaddleOCR
PaddleOCR is an open-source Optical Character Recognition (OCR) tool developed by PaddlePaddle, a deep learning platform from Baidu. 
It's a powerful library that supports text detection, recognition, and even layout analysis for a wide range of document types.
[Code](https://github.com/vvvvvvss/PaddleOCR)  
![image](https://github.com/user-attachments/assets/fadb1b11-693a-4d24-9ffd-41a696fa77db)
