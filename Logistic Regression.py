from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
seed_num = 100
print("---------------------------------------")
# Define functions
def evaluate_model(model, dataset, name):
    predictions = model.transform(dataset)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy of {name}: {accuracy}")
    return 

data_path = "file:/Workspace/Users/tan.xinxue@health.govt.nz/MAI/kdd.data"
try:

    df = (
        spark.read.format("csv")
        .option("header", "false")
        .option("inferSchema", "true")
        .load(data_path)
    )

    print("Data loaded successfully")

except Exception as e:
    print(f"Error loading data: {e}")
    raise
# Rename columns
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
df = df.toDF(*columns)
#display(df)

# Check null values
print("----------------------------------------")
print('Row count before dropping missing values:', df.count())

# Remove missing values
df = df.dropna()
print('Row count after dropping missing values:', df.count())
print("----------------------------------------")

# Identify categorical data type
# print(df.dtypes)
categorical_data_flag = False
for column in df.columns:
  if dict(df.dtypes)[column] == 'string':
    print(column)
    
    categorical_data_flag = True

if categorical_data_flag:
    print("Categorical data found")
else:
    print("No categorical data found")
print("---------------------------------------")
# Index the label column (convert string labels to numeric)
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
df = label_indexer.fit(df).transform(df)
#display(df)

# Assemble features into a vector
features_columns = df.columns
features_columns.remove('label')
features_columns.remove('label_index')

assembler = VectorAssembler(
    inputCols=features_columns,
    outputCol="features"
)

df_features = assembler.transform(df)

# reassemble the feature matrix with label index
df_input = df_features.select('features', 'label_index')
#df_input.show()

# Split data into training and test sets
trainingData, testData = df_input.randomSplit([0.8, 0.2], seed=seed_num)
print('Training data count:', trainingData.count())
print('Test data count:', testData.count())

# Train Logistic Regression 
lr = LogisticRegression(featuresCol='features', labelCol='label_index',maxIter=10,regParam=0.0)
lr_model = lr.fit(trainingData)
print("---------------------------------------")
# Evaluate model performance
# Evaluate with training data
evaluate_model(lr_model, trainingData, "Training Data")

# Evaluate with test data
evaluate_model(lr_model, testData, "Test Data")
