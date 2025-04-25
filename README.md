
# Analysis of Iris Dataset with PySpark and Logistic Regression

This notebook demonstrates a classification analysis of the Iris dataset using PySpark and Logistic Regression. The analysis includes data loading, preprocessing, model training, and evaluation.

## Problem Statements

1.  **Data Loading and Preparation:**
    * **Definition:** The first problem involves loading the Iris dataset into a PySpark DataFrame and preparing it for machine learning. This includes converting the dataset into a suitable format for PySpark's machine learning algorithms.
    * **Points:**
        * Load the Iris dataset using `sklearn.datasets.load_iris()`.
        * Convert the dataset into a Pandas DataFrame.
        * Create a PySpark DataFrame from the Pandas DataFrame.
        * Assemble the feature columns into a single vector column named "features" using `VectorAssembler`.

2.  **Model Training:**
    * **Definition:** The second problem focuses on training a Logistic Regression model using the prepared Iris dataset. This involves splitting the data into training and testing sets and fitting the model to the training data.
    * **Points:**
        * Split the dataset into training and testing sets using `randomSplit()`.
        * Initialize a `LogisticRegression` model, specifying the feature and label columns.
        * Train the Logistic Regression model using the training data.

3.  **Model Evaluation:**
    * **Definition:** The third problem is to evaluate the trained Logistic Regression model's performance on the test dataset. This involves making predictions and calculating the accuracy of the model.
    * **Points:**
        * Use the trained model to make predictions on the test data.
        * Evaluate the model's accuracy using `MulticlassClassificationEvaluator`.
        * Print the accuracy of the model.

## Code Explanation

### 1. Data Loading and Preparation

```python
from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder \
        .appName("ClassificationModel") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session initialized successfully.")
except Exception as e:
    print(f"Error initializing Spark session: {e}")
    exit(1)

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target

df = spark.createDataFrame(iris_df)
df.show(5)

from pyspark.ml.feature import VectorAssembler, StringIndexer

feature_cols = iris.feature_names
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df).select("features", "label")
df_assembled.show(5)
```

* A Spark session is initialized.
* The Iris dataset is loaded using `load_iris()` and converted to a Pandas DataFrame.
* The Pandas DataFrame is converted to a PySpark DataFrame.
* `VectorAssembler` is used to combine the feature columns into a single "features" column.

### 2. Model Training

```python
from pyspark.ml.classification import LogisticRegression

train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train_data)
```

* The data is split into training (80%) and testing (20%) sets.
* A `LogisticRegression` model is initialized.
* The model is trained using the training data.

### 3. Model Evaluation

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = lr_model.transform(test_data)
predictions.select("features", "label", "prediction").show(20)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")
```

* Predictions are made on the test data using the trained model.
* `MulticlassClassificationEvaluator` is used to calculate the accuracy of the predictions.
* The accuracy is printed to the console.
