# Spark ML Model Implementations

This repository showcases the implementation of several machine learning models using Apache Spark. Below is a summary of the models:

## Model Summary

### 1. Classification Model


* **Description:**
    * This notebook demonstrates a classification model built with Spark.
    * The Iris dataset is used for classification.
    * Logistic Regression is employed as the classification algorithm.
* **Key Procedures:**
    * Initialization of a Spark session.
    * Loading and preprocessing of the dataset, including feature vectorization.
    * Splitting the data into training and testing sets.
    * Training the Logistic Regression model.
    * Generating predictions on the test set.
    * Evaluating the model's accuracy.

### 2. Clustering Model


* **Description:**
    * This notebook demonstrates a clustering model built with Spark.
    * The Iris dataset (without the target column) is used for clustering.
    * K-Means clustering is the algorithm used.
* **Key Procedures:**
    * Initialization of a Spark session.
    * Loading the dataset and assembling features into a vector.
    * Training the K-Means clustering model.
    * Predicting cluster assignments for the data points.
    * Evaluating the clustering using the Silhouette Score.
    * Displaying the centers of the identified clusters.

### 3. Recommendation Engine


* **Description:**
    * This notebook demonstrates a recommendation engine built with Spark.
    * The MovieLens 20M dataset is used.
    * The Alternating Least Squares (ALS) algorithm is used for collaborative filtering.
* **Key Procedures:**
    * Initialization of a Spark session.
    * Loading and preprocessing the ratings data.
    * Splitting the data into training and testing sets.
    * Training the ALS recommendation model.
    * Generating predictions for movie ratings.
    * Evaluating the model using Root-Mean-Square Error (RMSE).
    * Generating movie recommendations for users and user recommendations for movies.
