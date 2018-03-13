package at.snn.util.data

import at.snn.model.data.SampleSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.util.Random

/**
  *
  */
object DataSampler {

  def sample(x: INDArray, y: INDArray, amount: Int = 100): (INDArray, INDArray) = {
    val xSample = Nd4j.createUninitialized(amount, x.columns())
    val ySample = Nd4j.createUninitialized(y.rows(), amount)
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
    val features = x.columns()
    val labels = y.rows()
    val trainRows = math.round((trainPercent / 100D) * total).toInt
    val cvRows = math.round((cvPercent / 100D) * total).toInt

    val idxShuffled = Random.shuffle((0 until x.rows()).toList)

    val (trainIdx, rest) = idxShuffled.splitAt(trainRows)
    val (cvIdx, testIdx) = rest.splitAt(cvRows)

    val trainingSet: INDArray = Nd4j.createUninitialized(trainRows, features)
    val trainingResultSet = Nd4j.createUninitialized(labels, trainRows)
    trainIdx.zipWithIndex.foreach { idx =>
      trainingSet(idx._2, ->) = x(idx._1, ->)
      trainingResultSet(->, idx._2) = y(->, idx._1)
    }

    val cvSet: INDArray = Nd4j.createUninitialized(cvRows, features)
    val cvResultSet = Nd4j.createUninitialized(labels, cvRows)
    cvIdx.zipWithIndex.foreach { idx =>
      cvSet(idx._2, ->) = x(idx._1, ->)
      cvResultSet(->, idx._2) = y(->, idx._1)
    }

    val testSet: INDArray = Nd4j.createUninitialized(testIdx.length, features)
    val testResultSet = Nd4j.createUninitialized(labels, testIdx.length)
    testIdx.zipWithIndex.foreach { idx =>
      testSet(idx._2, ->) = x(idx._1, ->)
      testResultSet(->, idx._2) = y(->, idx._1)
    }

    SampleSet(trainingSet, trainingResultSet, cvSet, cvResultSet, testSet, testResultSet)
  }

}
