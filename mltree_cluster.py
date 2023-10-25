# MR Tree in cluster configuration

# Dependencies
import time
import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark import SparkContext

def run(num_nodes, num_cpu):

    # --------- Start Spark Context ---------
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    # --------- Start Spark Session ---------
    spark = SparkSession.builder.getOrCreate()

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

    # Split into test/train
    (df_train_e, df_test) = df_assembled.randomSplit([0.7, 0.3])

    # --------- Train Model ---------

    # Init dataframe
    df_train = df_assembled.alias('df_train')

    # Start print sequence
    sys.stdout.write('m,dt,accuracy\n')

    # Init variables
    max_m = 600
    step = 30

    # Clear save file
    path = 'out/ml_tree_' + str(max_m) + '_' + str(num_nodes) + '_' + str(num_cpu) + '.csv'
    with open(path, 'w') as f:
        f.write('m,dt\n')

    for i in range(0, max_m, step):
        # Create the DecisionTree model
        tree = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='features')

        # Fit the model to the data and calculate performance
        dt = time.time()
        model = tree.fit(df_train)
        dt = time.time() - dt

        # Test model - predictions
        predictions = model.transform(df_test)
        # Test model - accuracy
        accuracy = predictions.filter(predictions.indexedLabel == predictions.prediction).count() / float(predictions.count())

        # Print info
        msg = str(i + 1) + "," + str(dt) + "," + str(accuracy) + "\n"
        sys.stdout.write(msg)

        # Append data to file
        with open(path, 'a') as f:
            f.write(msg)

        # Add data to train for next loop
        for _ in range(step):
            df_train = df_train.union(df_train_e)

    spark.stop()

sys.stdout.write('Script Open')
if __name__ == '__main__':
    # Catch inputs
    if len(sys.argv) > 2 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        sys.stdout.write('Running with ' + sys.argv[1] + ' nodes and ' + sys.argv[2] + ' CPUs')
        run(sys.argv[1], sys.argv[2])
    else:
        sys.stdout.write('Usage: python mltree.py num_nodes num_cpu')