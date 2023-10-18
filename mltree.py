# MR Tree in local configuration

# Dependencies
import time

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import DecisionTreeClassifier

# --------- Start Spark Session ---------
spark = SparkSession.builder.master("local[2]").appName("MySparkApp").getOrCreate()

# --------- Read Data ---------
# Read data into Spark DataFrame
df = spark.read.csv('data_a/adult.data', header=False, inferSchema=True)

# Fill NA
df = df.fillna(0)

# Define the features and label columns
feature_cols = df.columns[:-1]
label_col = df.columns[-1]

# --------- Data Preprocessing ---------
# 1] Get enmbeddings

# String indexers for categorical columns
str_cols = [col for col in feature_cols if df.select(col).dtypes[0][1] == 'string']
feature_indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in str_cols]

#   Get new feature column names
feature_cols_indexed = [indexer.getOutputCol() for indexer in feature_indexers] + [col for col in feature_cols if col not in str_cols]

# String indexer for label
labelIndexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

# Transform dataframe
df_indexed = Pipeline(stages=feature_indexers+[labelIndexer]).fit(df).transform(df)

# Set all columns to integer type
for column_name in feature_cols_indexed + ["indexedLabel"]:
    df_indexed = df_indexed.withColumn(column_name, col(column_name).cast(IntegerType()))

# 2] Get feature vector

# Vector assembler
assembler = VectorAssembler(inputCols=feature_cols_indexed, outputCol="features")

# Transform dataframe
df_assembled = assembler.transform(df_indexed).select("features", "indexedLabel")

# --------- Train Model ---------

# Init dataframe
df_train = df_assembled.alias('df_train')

# Init variables
performances = []
max_m = 201
step = 10

for i in range(0, max_m, step):
    # Create the DecisionTree model
    tree = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='features')

    # Fit the model to the data and calculate performance
    dt = time.time()
    model = tree.fit(df_train)
    dt = time.time() - dt

    # Add performance to list
    performances.append((i + 1, dt))

    # Add data to train for next loop
    for _ in range(step):
        df_train = df_train.union(df_assembled)

# --------- Save Results ---------
# Save to csv
with open('out/ml_tree_local.csv', 'w') as f:
    # Write header
    f.write('m,dt\n')

    # Write data
    for m, dt in performances:
        f.write(f'{m},{dt}\n')