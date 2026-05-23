#   spark-submit hdfs:///user/gaochen1/A3CodeAIO.py hdfs:///user/gaochen1/kdd.data hdfs:///user/gaochen1/results

import sys
import time
from math import sqrt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier,LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SEEDS = [1,2,3,4,5,6,7,8,9,10]

COLUMNS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
FEATURE_COLS = [c for c in COLUMNS if c != "label"]

def loadData(spark,inputPath):
    df = spark.read.format("csv") \
    .option("header", "false") \
    .option("inferSchema", "true") \
    .load(inputPath)

    # Rename columns
    df = df.toDF(*COLUMNS)
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
    for column in FEATURE_COLS:
        if dict(df.dtypes)[column] == 'string':
            print(column)
            
            categorical_data_flag = True

    if categorical_data_flag:
        print("Categorical data found")
    else:
        print("No categorical data found")
    print("---------------------------------------")

    # normal->0 anomaly->1
    df = df.withColumn(
        "labelIndex",
        when(col("label").startswith("normal"), 0.0).otherwise(1.0)
    )

    return df

def mainCycle(evaluator,pipeline,data,results):
    # main cycle
    for runID, seed in enumerate(SEEDS, start=1):

        #
        startTime = time.time()

        #data spliting
        trainDf, testDf = data.randomSplit([0.7, 0.3], seed=seed)
        trainDf = trainDf.cache()
        testDf = testDf.cache()


        model = pipeline.fit(trainDf)
        trainPred = model.transform(trainDf)
        testPred = model.transform(testDf)
        trainAccuracy = evaluator.evaluate(trainPred)
        testAccuracy = evaluator.evaluate(testPred)

        elapsed = time.time() - startTime
        #


        row = {
            "run": runID,
            "seed": seed,
            "trainAccuracy": trainAccuracy,
            "testAccuracy": testAccuracy,
            "runningTime": elapsed,
        }
        results.append(row)

        # print(
        #     f"Run {row['run']:02d} seed={row['seed']} "
        #     f"trainAcc={row['trainAccuracy']:.6f} "
        #     f"testAcc={row['testAccuracy']:.6f} "
        #     f"time={row['runningTime']:.2f}s"
        # )

        trainDf.unpersist()
        testDf.unpersist()

def output(spark,results,outputPath,suffix):
    outputPath=f"{outputPath}/{suffix}"

    trainAccuracies = [r["trainAccuracy"] for r in results]
    testAccuracies = [r["testAccuracy"] for r in results]
    runtimes = [r["runningTime"] for r in results]

    print(f"\n========== {suffix} Summary 10 Runs ==========")

    print(f"Train Accuracy: max={max(trainAccuracies):.6f}, min={min(trainAccuracies):.6f}, "
          f"avg={sum(trainAccuracies)/len(trainAccuracies):.6f}, std={StdDev(trainAccuracies):.6f}")
    
    print(f"Test Accuracy : max={max(testAccuracies):.6f}, min={min(testAccuracies):.6f}, "
          f"avg={sum(testAccuracies)/len(testAccuracies):.6f}, std={StdDev(testAccuracies):.6f}")
    
    print(f"Running Time sec   : max={max(runtimes):.2f}, min={min(runtimes):.2f}, "
          f"avg={sum(runtimes)/len(runtimes):.2f}, std={StdDev(runtimes):.2f}")
    
    print("\n")

    resultsDf = spark.createDataFrame(results)
    (
        resultsDf.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(outputPath)
    )
    print(f"Saved {suffix} results to: {outputPath}")

def StdDev(values):
    meanValue = sum(values) / len(values)
    return sqrt(sum((x - meanValue) ** 2 for x in values) / len(values))


def main(inputPath,outputPath):
    # init spark
    spark = (SparkSession.builder.appName("AIML427AIO").getOrCreate())

    #load&preprocess
    data = loadData(spark, inputPath).cache()
    
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="labelIndex",
        predictionCol="prediction",
        seed=1
    )
    pipelineDT = Pipeline(stages=[assembler, dt])
    resultsDT = []

    lr = LogisticRegression(
        featuresCol='features', 
        labelCol='labelIndex',
        predictionCol="prediction",
        maxIter=10,
        regParam=0.0
    )
    pipelineLR = Pipeline(stages=[assembler, lr])
    resultsLR = []

    evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction",metricName="accuracy")

    #
    mainCycle(evaluator,pipelineDT,data,resultsDT)
    mainCycle(evaluator,pipelineLR,data,resultsLR)

    #
    output(spark,resultsDT,outputPath,"DecisionTree")
    output(spark,resultsLR,outputPath,"LogisticRegression")

    data.unpersist()
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        # default value
        sys.argv = [
            "A3CodeAIO.py",
            "kdd.data",
            "results"
        ]
        # print("Usage: spark-submit A3CodeAIO.py <inputPath> <outputPath>", file=sys.stderr)
        # sys.exit(1)

    main(sys.argv[1], sys.argv[2])
