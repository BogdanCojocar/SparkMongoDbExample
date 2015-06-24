package com.spark.classification.data

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ HashingTF, Tokenizer }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{ Row, SQLContext }
import org.apache.spark.ml.evaluation.Evaluator

/**
 * @author boco8775
 */
case class Document(id: Long, value: String)

class SparkClassifierEstimation(sparkContext: SparkContext) {

  val sqlContext = new SQLContext(sparkContext)
  import sqlContext.implicits._

  def evaluate = {

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("value")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(3)

    // Run cross-validation, and choose the best set of parameters.
    val repository = new MongoDbRepository(sparkContext)
    val cvModel = crossval.fit(repository.queryData)

    // Prepare test documents, which are unlabeled.
    val test = sparkContext.parallelize(Seq(
      Document(4L, "spark i j k"),
      Document(5L, "l m n"),
      Document(6L, "mapreduce spark"),
      Document(7L, "apache hadoop")))

    cvModel.transform(test.toDF())
      .select("id", "value", "probability", "prediction")
      .collect()
      .foreach {
        case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
          println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }
  }
}