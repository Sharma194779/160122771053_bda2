

**1. Classification Model with Spark (Iris Dataset)**

```markdown
#   Classification Model with Spark (Iris Dataset)

This notebook performs a classification analysis using PySpark and Logistic Regression on the Iris dataset.

##   Problem Statements

1.  **Data Loading and Preparation:**
    * **Definition:** Load the Iris dataset, convert it into a PySpark DataFrame, and prepare it for machine learning by assembling features into a single vector column.
    * **Points:**
        * Load the Iris dataset using `sklearn.datasets.load_iris()`.
        * Convert the dataset to a Pandas DataFrame.
        * Create a PySpark DataFrame from the Pandas DataFrame.
        * Use `VectorAssembler` to combine feature columns into a "features" column.

2.  **Model Training:**
    * **Definition:** Train a Logistic Regression model on the prepared Iris dataset by splitting the data into training and testing sets and fitting the model.
    * **Points:**
        * Split the dataset into training and testing sets using `randomSplit()`.
        * Initialize a `LogisticRegression` model, specifying feature and label columns.
        * Train the model with the training data.

3.  **Model Evaluation:**
    * **Definition:** Evaluate the trained model's performance by making predictions on the test set and calculating the accuracy.
    * **Points:**
        * Generate predictions on the test data using the trained model.
        * Calculate the model's accuracy using `MulticlassClassificationEvaluator`.
        * Display the accuracy.

##   Code Explanation

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

train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = lr_model.transform(test_data)
predictions.select("features", "label", "prediction").show(20)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")
```

```
```

**2. Clustering Model with Spark (Iris Dataset)**

```markdown
#   Clustering Model with Spark (Iris Dataset)

This notebook performs clustering on the Iris dataset using PySpark and KMeans.

##   Problem Statements

1.  **Data Loading and Preparation:**
    * **Definition:** Load the Iris dataset and convert it into a PySpark DataFrame, preparing it for clustering.
    * **Points:**
        * Load the Iris dataset using `sklearn.datasets.load_iris()`.
        * Convert the dataset to a Pandas DataFrame.
        * Create a PySpark DataFrame.

2.  **Feature Assembling:**
    * **Definition:** Assemble the feature columns of the Iris dataset into a single vector column suitable for the KMeans algorithm.
    * **Points:**
        * Use `VectorAssembler` to combine feature columns into a "features" column.

3.  **Model Training and Prediction:**
    * **Definition:** Train a KMeans clustering model and predict cluster assignments for the Iris data points.
    * **Points:**
        * Initialize a `KMeans` model, specifying the number of clusters (k=3 for Iris).
        * Train the KMeans model.
        * Generate cluster predictions.

4.  **Model Evaluation:**
    * **Definition:** Evaluate the quality of the clustering using the Silhouette Score.
    * **Points:**
        * Use `ClusteringEvaluator` to calculate the Silhouette Score.
        * Print the Silhouette Score.

5.  **Cluster Centers:**
    * **Definition:** Display the center of each identified cluster.
    * **Points:**
        * Retrieve the cluster centers from the trained KMeans model.
        * Print the coordinates of each cluster center.

##   Code Explanation

```python
from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder \
        .appName("ClusteringModel") \
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

df = spark.createDataFrame(iris_df)
df.show(5)

from pyspark.ml.feature import VectorAssembler

feature_cols = iris.feature_names
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df).select("features")
df_assembled.show(5)

from pyspark.ml.clustering import KMeans

kmeans = KMeans(featuresCol='features', k=3, seed=1)  # 3 clusters for iris
model = kmeans.fit(df_assembled)

predictions = model.transform(df_assembled)
predictions.select("features", "prediction").show(10)

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.4f}")

centers = model.clusterCenters()
for idx, center in enumerate(centers):
    print(f"Cluster {idx} Center: {center}")

spark.stop()
print("Spark session stopped successfully.")
```

```
```

**3.Recommendation Engine with Spark (Movie Ratings)**

```markdown
#   Recommendation Engine with Spark (Movie Ratings)

This notebook builds a movie recommendation engine using PySpark and the Alternating Least Squares (ALS) algorithm.

##   Problem Statements

1.  **Data Loading and Preparation:**
    * **Definition:** Load the movie ratings dataset into PySpark DataFrames.
    * **Points:**
        * Load the `ratings.csv` and `movies.csv` datasets.
        * Display a preview of the ratings data.

2.  **Data Preprocessing:**
    * **Definition:** Select relevant columns and cast them to the appropriate data types.
    * **Points:**
        * Select "userId", "movieId", and "rating" columns from the ratings data.
        * Cast "userId" and "movieId" to integer type.
        * Cast "rating" to float type.

3.  **Model Training:**
    * **Definition:** Train a recommendation model using the ALS algorithm.
    * **Points:**
        * Split the data into training and testing sets.
        * Initialize an `ALS` model with specified parameters (rank, maxIter, regParam, coldStartStrategy).
        * Train the ALS model using the training data.

4.  **Prediction and Evaluation:**
    * **Definition:** Make rating predictions on the test data and evaluate the model's performance.
    * **Points:**
        * Generate rating predictions using the trained model.
        * Evaluate the model using Root Mean Square Error (RMSE).
        * Print the RMSE.

5.  **Generate Recommendations:**
    * **Definition:** Generate movie recommendations for users and user recommendations for movies.
    * **Points:**
        * Generate top-N movie recommendations for all users.
        * Generate top-N user recommendations for all movies.
        * Display the recommendations.

##   Code Explanation

```python
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("RecommendationEngine").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").getOrCreate()

ratings = spark.read.csv("ml-20m/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("ml-20m/movies.csv", header=True, inferSchema=True)

ratings.show(5)

ratings = ratings.select("userId", "movieId", "rating")
ratings = ratings.withColumn("userId", col("userId").cast("int"))
ratings = ratings.withColumn("movieId", col("movieId").cast("int"))
ratings = ratings.withColumn("rating", col("rating").cast("float"))

(training_data, test_data) = ratings.randomSplit([0.8, 0.2])

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=10, maxIter=10, regParam=0.1, coldStartStrategy="drop")
model = als.fit(training_data)

predictions = model.transform(test_data)
predictions.show(5)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-Mean-Square Error = {rmse}")

user_recommendations = model.recommendForAllUsers(5)
user_recommendations.show(5)

movie_recommendations = model.recommendForAllItems(5)
movie_recommendations.show(5)

spark.stop()
print("Spark session stopped successfully.")


These Markdown representations provide a clear structure for understanding the purpose, problems addressed, and implementation details of each notebook.
