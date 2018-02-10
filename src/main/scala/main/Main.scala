package main

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import util.{GradientChecker, MatlabImporter}

import scala.util.Random

/**
  *
  */
object Main {

  def main(args: Array[String]): Unit = {

    val map = MatlabImporter("src/main/resources/test.mat")
    println(map.keys)

    val x = map("X")
    val y = map("y")
    val yMappedCols = Nd4j.zeros(1, y.rows() * 10)
    for (i <- 0 until y.rows) {
      val yVal: Int = y(i, 0).intValue()
      val updIdx = (yVal % 10) + 10 * i
      yMappedCols(0, updIdx) = 1
    }

    val yReshaped = yMappedCols.reshape('f', 10, y.rows())

    val (xS, yS) = sample(x, yReshaped, 5)
    GradientChecker.check()
  }

  def sample(x: INDArray, y: INDArray, amount: Int = 100) = {
    val xSample = Nd4j.zeros(amount, x.columns())
    val ySample = Nd4j.zeros(y.rows(), amount)
    Random.shuffle((0 until x.rows()).toList)
      .take(amount)
      .zipWithIndex
      .foreach((iAndIdx) => {
        xSample(iAndIdx._2, ->) = x(iAndIdx._1, ->)
        ySample(->, iAndIdx._2) = y(->, iAndIdx._1)
      })
    (xSample, ySample)
  }

}