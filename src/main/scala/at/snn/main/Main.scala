package at.snn.main

import at.snn.util.GradientChecker
import at.snn.util.data.MatlabImporter
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.util.Random

/**
  *
  */
object Main {

  def main(args: Array[String]): Unit = {

    val rawData = MatlabImporter("src/main/resources/test.mat")

    val (x, yMappedCols) = rawData.getData()

    val (xS, yS) = sample(x, yMappedCols, 5)
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