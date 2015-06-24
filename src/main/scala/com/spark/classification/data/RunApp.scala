package com.spark.classification.data

import org.apache.spark.{SparkConf, SparkContext}

/**
 * @author boco8775
 */
object Test {
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("Spark Classification").setMaster("local[*]")
    val sparkContext = new SparkContext(conf)
    val classifEstimator = new SparkClassifierEstimation(sparkContext)
    classifEstimator.evaluate
    
    sparkContext.stop()
  }
}