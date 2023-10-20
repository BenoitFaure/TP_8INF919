# MR Tree in local configuration

# Dependencies
import time
import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark import SparkContext
sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")
job_id = sc.getConf().get("spark.app.id")

# --------- Start Spark Session ---------
spark = SparkSession.builder.getOrCreate() # SparkSession.builder.master("local[2]").appName('ml_tree').getOrCreate()

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
str_cols = [c for c in feature_cols if df.select(c).dtypes[0][1] == 'string']
feature_indexers = [StringIndexer(inputCol=c, outputCol=c+'_index') for c in str_cols]

#   Get new feature column names
feature_cols_indexed = [indexer.getOutputCol() for indexer in feature_indexers] + [c for c in feature_cols if c not in str_cols]

# String indexer for label
labelIndexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

# Transform dataframe
df_indexed = Pipeline(stages=feature_indexers+[labelIndexer]).fit(df).transform(df).select(feature_cols_indexed + ["indexedLabel"])

# # Set all columns to integer type
df_indexed = df_indexed.select([col(c).cast("integer") for c in df_indexed.columns])

# 2] Get feature vector

# Vector assembler
assembler = VectorAssembler(inputCols=feature_cols_indexed, outputCol="features")

# Transform dataframe
df_assembled = assembler.transform(df_indexed).select("features", "indexedLabel")

# --------- Train Model ---------

# Init dataframe
df_train = df_assembled.alias('df_train')

# Start print sequence
sys.stdout.write('m,dt\n')

# Init variables
performances = []
max_m = 151
step = 10

# Clear save file
path = 'out/ml_tree_' + str(job_id) + '_' + str(max_m) + '.csv'
with open(path, 'w') as f:
    f.write('m,dt\n')

for i in range(0, max_m, step):
    # Create the DecisionTree model
    tree = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='features')

    # Fit the model to the data and calculate performance
    dt = time.time()
    model = tree.fit(df_train)
    dt = time.time() - dt

    # Print info
    sys.stdout.write(str(i + 1) + "," + str(dt) + "\n")

    # Append data to file
    with open(path, 'a') as f:
        f.write(str(i + 1) + "," + str(dt) + "\n")

    # Add data to train for next loop
    for _ in range(step):
        df_train = df_train.union(df_assembled)

spark.stop()