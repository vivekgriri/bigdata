from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("MLlib FraudML Prediction").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'fraud_data' into a Spark DataFrame
fraud_df = spark.sql("SELECT * FROM fraud_data ")

# Step 3: Handle null values by either dropping or filling them
fraud_df = fraud_df.na.drop()  # Drop rows with null values

# Feature Engineering
# Selecting numerical columns for the model
feature_cols = ["transaction_amount", "account_balance", "daily_transaction_count",
                "avg_transaction_amount_7d", "risk_score", "transaction_distance"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(fraud_df).select("features", "Fraud_Label")

print(f'===================Feature engineering end====================')

# Split Data
train_data, test_data = final_data.randomSplit([0.7, 0.3])

# Random Forest Model
rf = RandomForestClassifier(labelCol="Fraud_Label", featuresCol="features", numTrees=50)
model = rf.fit(train_data)
print(f'================== RFM End======================================')
# Predictions and Evaluation

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="Fraud_Label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f'================== Predictions and Evaluation End======================================')
# Step 8: Print the model performance metrics
print(f"model_results:algorithm {'RandomForest'}")
print(f"model_results:auc_score: {str(auc).encode('utf-8')}")
print(f'================== Results end ======================================')

# ---- Write metrics to HBase with happybase (using the provided pattern) ----
# Example data (row_key, column_family:column, value) populated with the metrics
data = [
    ('metrics1', 'model_results:algorithm', 'RandomForest'),
    ('metrics1', 'model_results:auc_score', str(auc).encode('utf-8')),
]


# Function to write data to HBase inside each partition
def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('fraud_metrics')  # Update table name
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()


# Parallelize data and apply the function with foreachPartition
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop the Spark session
spark.stop()
