How to install and run

Below are the steps on how to run the program on the school's Hadoop cluster. 
We connect to the cluster, set up the Spark environment, put the data on HDFS, and submit the program. 
Replace $USER_NAME with your own ECS user name.

Step 1. Connect to the cluster with two SSH hops.

```
ssh $USER_NAME@barretts.ecs.vuw.ac.nz
ssh co246a-1
```

After the second hop the prompt changes to co246a-1%.

Step 2. Set up the Spark environment. Copy SetupSparkClasspath.sh into your working directory and source it.

```
source SetupSparkClasspath.sh
```

Nothing is printed, so we verify it worked.

```
echo $HADOOP_VERSION
echo $PATH
```

Step 3. Get a Kerberos ticket so that HDFS works. The ticket lasts for the day, so we do this again each session.

```
kinit $USER_NAME
```

Step 4. Put the input data on HDFS. The program reads from HDFS, so kdd.data must be there first.

```
hdfs dfs -put kdd.data /user/$USER_NAME/kdd.data
```

(HDFS commands can be used:)

```
hdfs dfs -ls /user/$USER_NAME     list your files
hdfs dfs -put filename path       upload a file
hdfs dfs -get filename            download a file
hdfs dfs -rm filename             delete a file
```

Step 5. Submit the program with spark-submit. We run it on YARN in cluster mode. We pass the JAVA_HOME for the cluster containers, the program file, the input path, and the output path.

```
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.yarn.appMasterEnv.JAVA_HOME=/usr/lib/jvm/java-21-openjdk \
  --conf spark.executorEnv.JAVA_HOME=/usr/lib/jvm/java-21-openjdk \
  A3CodeAIO.py \
  hdfs:///user/$USER_NAME/kdd.data \
  hdfs:///user/$USER_NAME/results
```

The program file A3CodeAIO.py can be a local path. spark-submit uploads it to HDFS automatically.

Step 6. Check that the job succeeded. In the YARN log, look for the line that says final status: SUCCEEDED. The word FINISHED on its own does not mean success, so we always check the final status.

Step 7. Read the results from HDFS. In cluster mode the driver runs inside a YARN container, so anything the program prints does not come back to the terminal. The numbers are written to HDFS instead, so we read them from there. Wrap each path in single quotes because the shell may try to expand the star by itself.

```
hdfs dfs -cat '/user/$USER_NAME/results/DecisionTree/summary/part-*'
hdfs dfs -cat '/user/$USER_NAME/results/LogisticRegression/summary/part-*'
hdfs dfs -cat '/user/$USER_NAME/results/DecisionTree/runs/part-*'
hdfs dfs -cat '/user/$USER_NAME/results/LogisticRegression/runs/part-*'
```
