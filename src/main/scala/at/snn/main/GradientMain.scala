package at.snn.main

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.NNBuilder
import at.snn.util.data.{DataSampler, LabelConverter, MatlabImporter}
import at.snn.util.optimizers.{GradientDescendOptimizer, MomentumOptimizer, NesterovAcceleratedOptimizer}
import at.snn.util.plot.{ChartRenderer, PlotCost}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object GradientMain extends StrictLogging {

  def main(args: Array[String]): Unit = {
    Nd4j.setDataType(Type.DOUBLE)

    val map = MatlabImporter("src/main/resources/test.mat")
    val x = map("X")
    val y = map("y")
    val yMappedCols = Nd4j.zeros(10, y.rows())
    for (i <- 0 until y.rows) {
      val yVal: Int = y(i, 0).intValue()
      LabelConverter.labelToVector(yMappedCols(->, i), yVal, 10)
    }

    val inputsSource = x.columns()
    val labels = 10
    val iterations = 200
    val lambda = 4
    val learnRate = 1

    val inputLayer = NNBuilder.buildNetwork(inputsSource, labels, 35, 35, 35, 35)

    val dataset = DataSampler.createSampleSet(x, yMappedCols)

    val optimizers: Seq[(String, (INDArray, INDArray, InputLayer, Int, Double, Double) => Seq[Double])] =
      Seq(
        ("Gradient Descent", GradientDescendOptimizer.minimize),
        ("Momentum", MomentumOptimizer.minimize(_, _, _, _, _, _, 0.5)),
        ("Nesterov", NesterovAcceleratedOptimizer.minimize(_, _, _, _, _, _, 0.5))
      )

    val nameAndCosts = optimizers
      .map({ case (name, optimizer) =>
        val network = inputLayer.copyNetwork
        val costs = optimizer(dataset.trainingSet, dataset.trainingResultSet, network, iterations, lambda, learnRate)
        (name, network, costs)
      }).map({ case (name, network, costs) =>
      logger.info("Testing {} optimization:", name)
      runOnTrainingSet(network, dataset)
      runOnCvSet(network, dataset)
      (name, costs)
    })

    ChartRenderer.render(PlotCost.plot(nameAndCosts))
  }


  private def runOnCvSet(inputLayer: InputLayer, dataset: SampleSet): Unit = {
    val cvSet = dataset.cvSet
    val cvResultSet = dataset.cvResultSet
    var falseCount = 0
    for (i <- 0 until cvSet.rows()) {
      val result = inputLayer.activate(cvSet(i, ->))
      val y = cvResultSet(->, i)
      val yLabel = LabelConverter.vectorToLabel(y)
      val predictLabel = LabelConverter.vectorToLabel(result)
      if (yLabel != predictLabel) {
        debugFalsePrediction(result, y)
        falseCount += 1
      }
    }
    logger.info(s"CV Total predictions: ${cvSet.rows()}")
    logger.info(s"CV Incorrect predictions: ${falseCount}")
    logger.info(s"CV Correct predictions%: ${100 - (falseCount / cvSet.rows().toDouble) * 100D}")
    logger.info("---------------------\r\n\r\n")
  }

  private def runOnTrainingSet(inputLayer: InputLayer, dataset: SampleSet): Unit = {
    val tSet = dataset.trainingSet
    val tResultSet = dataset.trainingResultSet
    var falseCount = 0
    for (i <- 0 until tSet.rows()) {
      val result = inputLayer.activate(tSet(i, ->))
      val y = tResultSet(->, i)
      val yLabel = LabelConverter.vectorToLabel(y)
      val predictLabel = LabelConverter.vectorToLabel(result)
      if (yLabel != predictLabel) {
        debugFalsePrediction(result, y)
        falseCount += 1
      }
    }
    logger.info(s"Train Total predictions: ${tSet.rows()}")
    logger.info(s"Train Incorrect predictions: ${falseCount}")
    logger.info(s"Train Correct predictions%: ${100 - (falseCount / tSet.rows().toDouble) * 100D}")
  }

  def debugFalsePrediction(prediction: INDArray, y: INDArray): Unit = {
    logger.debug("\r\nPredicted[{}] as [{}]\r\nRealValue[{}] is [{}]", prediction, LabelConverter.vectorToLabel(prediction),
      y, LabelConverter.vectorToLabel(y))
  }
}