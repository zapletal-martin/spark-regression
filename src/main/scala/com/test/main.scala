/*
package com.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}

object SparkRegression extends App {
  override def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("Linear regression").setMaster("local"))
    Logger.getRootLogger.setLevel(Level.DEBUG)

    //MLUtils.loadLabeledPoints()

    val data = sc.parallelize(1 to 100)
      .map(x => LabeledPoint(
        x,
        Vectors.dense(
          Array[Double]((x.toDouble - 1d) / 100d, x.toDouble / 100d, (x.toDouble + 1d) / 100d))))

    val splits = data randomSplit Array(0.8, 0.2)

    val training = splits(0) cache
    val test = splits(1) cache

    val numTraining = training count
    val numTest = test count

    println(s"Training: $numTraining, test: $numTest.")

    val algorithm = new LinearRegressionWithSGD()
    algorithm
      .optimizer
        .setNumIterations(10)
        .setStepSize(1)
        //.setUpdater(new SquaredL2Updater())
        .setRegParam(0.1)

    val model = algorithm run training

    val prediction = model predict(test map(_ features))

    val predictionAndLabel = prediction zip(test map(_ label))

    println("RESULTS")
    predictionAndLabel.foreach((result) => println(s"p: ${result._1}, l: ${result._2}"))

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)

    val rmse = math.sqrt(loss / numTest)

    println(s"Test RMSE = $rmse.")

    sc.stop()
  }
}
*/
