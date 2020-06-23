#!/home/coinlab/anaconda2/envs/spark_env/bin spark_env
# -*- coding: utf-8 -*-
import pickle
from functools import reduce
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.sql.functions import col, when, rand, lit


class DetectorWithRegression:
    def __init__(self, data, sc):
        self.data = data
        self.sc = sc

    def fit(self):
        """
        Creates the pipeline, splits the data , fits the model and save the model, also evaluates the results
        :return:
        """
        cols = [x for x in self.data.columns if x not in ['datetime', 'label', 'speed_overground']]
        assembler = VectorAssembler(handleInvalid="keep").setInputCols \
            (cols).setOutputCol("features")

        print('assembler')
        train = assembler.transform(self.data)
        train = train.drop(*cols)
        gbt = GBTRegressor(labelCol="speed_overground", featuresCol="features", predictionCol='predictions')

        print('Train model.  This also runs the indexers.')
        model = gbt.fit(train)
        # Save and load model
        model.write().overwrite().save('myGBTRegressor_nan')
        predictions = model.transform(train)
        evaluator = RegressionEvaluator(
            labelCol="speed_overground", predictionCol="predictions", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    def predict(self):
        """

        :return:
        """
        cols = [x for x in self.data.columns if x not in ['datetime', 'label']]
        assembler = VectorAssembler(handleInvalid="keep").setInputCols \
            (cols).setOutputCol("features")

        print('assembler')
        test = assembler.transform(self.data)
        test = test.drop(*cols)

        rf = GBTRegressionModel.load('myGBTRegressor_nan')
        preds = rf.transform(test)
        print(preds.printSchema())

        preds.write.save("regression_preds_5.parquet")

        return preds
