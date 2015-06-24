package com.spark.classification.data

import com.mongodb.casbah.{ WriteConcern => MongodbWriteConcern }
import com.stratio.provider._
import com.stratio.provider.mongodb._
import com.stratio.provider.mongodb.schema._
import com.stratio.provider.mongodb.writer._
import org.apache.spark.sql._
import DeepConfig._
import MongodbConfig._
import org.apache.spark._

/**
 * @author boco8775
 */
object MongoDbRepository {
  val DATABASE = "local"
  val COLLECTION = "test"
  val HOST = "localhost:27017"
}

import MongoDbRepository._

class MongoDbRepository(sparkContext: SparkContext) {
  val builder = MongodbConfigBuilder(Map(
    Host -> List(HOST),
    Database -> DATABASE,
    Collection -> COLLECTION,
    SamplingRatio -> 1.0,
    WriteConcern -> MongodbWriteConcern.Normal))
    
  val readConfig = builder.build()
  val sqlContext = new SQLContext(sparkContext)
  val mongoRDD = sqlContext.fromMongoDB(readConfig)
  mongoRDD.registerTempTable(COLLECTION)
  
  def queryData: DataFrame = {
    sqlContext.sql("SELECT value, label FROM " + COLLECTION)
  }
}