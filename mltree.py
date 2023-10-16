# Dependencies
import numpy as np
from pyspark.sql import SparkSession

# ---------DATA PROCESSING---------
# Load data
data = np.genfromtxt('data_a/adult.data', delimiter=',', dtype=str)


spark = SparkSession.builder.master("local[2]").appName("MySparkApp").getOrCreate()

# Read data into Spark DataFrame
df = spark.read.csv('data_a/adult.data', header=False, inferSchema=True)

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Define the features and label columns
feature_cols = df.columns[:-1]
label_col = df.columns[-1]


# Create a list of StringIndexer objects for string columns
indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in feature_cols if df.select(col).dtypes[0][1] == 'string']

# Create a Pipeline object to chain together the indexers
pipeline = Pipeline(stages=indexers)

# Fit the Pipeline to the data
df = pipeline.fit(df).transform(df)


# Define the integer columns
str_cols = [indexer.getInputCol() for indexer in indexers]
int_cols = [col for col in feature_cols if col not in str_cols]


# Create a VectorAssembler object to combine the feature columns into a single vector
assembler = VectorAssembler(inputCols=[col+'_index' for col in str_cols] + int_cols, outputCol='features')

# Transform the DataFrame to include the features column
df = assembler.transform(df).select(label_col, 'features')


# Create the DecisionTree model
dt = DecisionTreeClassifier(labelCol=label_col, featuresCol='features')

# Fit the model to the data
model = dt.fit(df)



spark.stop()

exit()