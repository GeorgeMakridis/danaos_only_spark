import datetime
from pyspark.sql import SparkSession
from demo_b import DetectorLevel_1

spark = SparkSession.builder.appName('ml-danaos').config("'spark.debug.maxToStringFields", 1000). \
    config("spark.hadoop.fs.s3a.endpoint", "http://127.0.0.1:9000"). \
    config("spark.hadoop.fs.s3a.access.key", "minio"). \
    config("spark.hadoop.fs.s3a.secret.key", "miniominio"). \
    config("spark.hadoop.fs.s3a.path.style.access", True). \
    config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"). \
    config("spark.driver.memory", "30g").config('spark.executor.memory', '12G') \
    .config('spark.storage.memoryFraction', '0').config('spark.driver.maxResultSize', '12g').config(
    'spark.kryoserializer.buffer.max', '1999m') \
    .config('spark.debug.maxToStringFields', '100'). \
    master("local[*]").getOrCreate()

spark.sparkContext.addPyFile("jars/aws-java-sdk-1.11.534.jar")
spark.sparkContext.addPyFile("jars/aws-java-sdk-core-1.11.534.jar")
spark.sparkContext.addPyFile("jars/aws-java-sdk-dynamodb-1.11.534.jar")
spark.sparkContext.addPyFile("jars/aws-java-sdk-kms-1.11.534.jar")
spark.sparkContext.addPyFile("jars/aws-java-sdk-s3-1.11.534.jar")
spark.sparkContext.addPyFile("jars/hadoop-aws-3.1.2.jar")
spark.sparkContext.addPyFile("jars/httpclient-4.5.3.jar")
spark.sparkContext.addPyFile("jars/joda-time-2.9.9.jar")

if __name__ == '__main__':

    print(datetime.datetime.now())
    detector_1 = DetectorLevel_1.DetectorLevel_1(param='message', sc=spark)
    # detector_1.run_xgb_training('XGB training started..')
    detector_1.run_xgb_prediction('XGB prediction started..')
