package util

import model.data.SampleSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.util.Random

/**
  *
  */
object DataSampler {

  def sample(x: INDArray, y: INDArray, amount: Int = 100): (INDArray, INDArray) = {
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

  def createSampleSet(x: INDArray, y: INDArray, trainPercent: Int = 60, cvPercent: Int = 20, testPercent: Int = 20): SampleSet = {
    require(trainPercent + cvPercent + testPercent == 100)
    val total = x.rows()
    val trainRows = math.round((trainPercent / 100D) * total).toInt
    val cvRows = math.round((cvPercent / 100D) * total).toInt

    val xShuffled = x.dup()
    val yShuffled = y.dup()
    Nd4j.shuffle(xShuffled)
    Nd4j.shuffle(yShuffled)

    val trainingSet = xShuffled(0 until trainRows, ->)
    val cvSet = xShuffled(trainRows until trainRows + cvRows, ->)
    val testSet = xShuffled(trainRows + cvRows until total, ->)
    val trainingResultSet = yShuffled(->, 0 until trainRows)
    val cvResultSet = yShuffled(->, trainRows until trainRows + cvRows)
    val testResultSet = yShuffled(->, trainRows + cvRows until total)
    SampleSet(trainingSet, trainingResultSet, cvSet, cvResultSet, testSet, testResultSet)
  }

}
