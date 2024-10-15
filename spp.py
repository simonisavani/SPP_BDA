from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import SparkConf, SparkContext

conf = SparkConf().set("spark.executor.extraClassPath", "C:/Users/91937/OneDrive/Desktop/SPP_BDA/SPP_BDA/Lib") \
                  .set("spark.driver.extraClassPath", "C:/Users/91937/OneDrive/Desktop/SPP_BDA/SPP_BDA/Lib")
sc = SparkContext(conf=conf)


spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()


df = spark.read.csv('C:/Users/91937/OneDrive/Desktop/SPP_BDA/SPP_BDA/NFLX.csv', header=True, inferSchema=True)

df.show(5)


assembler = VectorAssembler(inputCols=['Open', 'High', 'Low', 'Volume'], outputCol='features')
df = assembler.transform(df)


data = df.select(col('features'), col('Close').alias('label'))


train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol='features', labelCol='label', regParam=0.1, maxIter=100, fitIntercept=True)

lr_model = lr.fit(train_data)

predictions = lr_model.transform(test_data)


evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='mse')
mse = evaluator.evaluate(predictions)
print(f'Mean Squared Error: {mse}')


predictions.select('label', 'prediction').show(5)


spark.stop()

# import numpy as np
# np.show_config()
