package at.snn.main

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.data._
import at.snn.util.optimizers.{GradientDescendOptimizer, MomentumOptimizer, NesterovAcceleratedOptimizer}
import at.snn.util.plot.{ChartRenderer, PlotCost}
import at.snn.util.{NNBuilder, NNRunner}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.StdIn

/**
  *
  */
object GradientMain extends StrictLogging {

  def main(args: Array[String]): Unit = {
    Nd4j.setDataType(Type.DOUBLE)

    println("Please choose the input source:\r\n" +
      "1: Number recognition\r\n" +
      "2: Red wine quality\r\n" +
      "3: White wine quality\r\n" +
      "4: Diabetes data")
    val choice = StdIn.readInt()

    val rawData = choice match {
      case 1 => MatlabImporter("src/main/resources/numbers.mat")
      case 2 => WineImporter("src/main/resources/wine/winequality-red.csv")
      case 3 => WineImporter("src/main/resources/wine/winequality-white.csv")
      case 4 => DiabetesImporter("src/main/resources/diabetes/diabetes.csv")
    }

    val (x, yMappedCols) = rawData.getData
    val inputsSource = x.columns()
    val labels = rawData.labels
    val iterations = 200
    val lambda = 4
    val learnRate = 1

    println("Use default neuronal network[enter], or insert layer sizes separated by commas.")
    val layers: Array[Int] = StdIn.readLine()
      .split(',')
      .map(_.trim)
      .withFilter(_.isEmpty == false)
      .map(_.toInt)
    match {
      case Array() => Array(((labels + inputsSource) / 2.0).ceil.toInt)
      case empt@Array(_*) => empt
    }

    val inputLayer = NNBuilder.buildNetwork(inputsSource, labels, layers: _*)

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
      logger.info("\r\nTesting {} optimization:", name)
      runOnTrainingSet(network, dataset)
      runOnCvSet(network, dataset)
      (name, costs)
    })

    ChartRenderer.render(PlotCost.plot(nameAndCosts))
  }


  private def runOnCvSet(inputLayer: InputLayer, dataset: SampleSet): Unit = {
    val result = NNRunner.runPredictions(inputLayer, dataset.cvSet, dataset.cvResultSet)
    logger.info(s"CV Total predictions: ${result.totalPredictions}")
    logger.info(s"CV Incorrect predictions: ${result.wrongPredictions}")
    logger.info(s"CV Correct predictions%: ${result.correctPercent}")
  }

  private def runOnTrainingSet(inputLayer: InputLayer, dataset: SampleSet): Unit = {
    val result = NNRunner.runPredictions(inputLayer, dataset.trainingSet, dataset.trainingResultSet)
    logger.info(s"Train Total predictions: ${result.totalPredictions}")
    logger.info(s"Train Incorrect predictions: ${result.wrongPredictions}")
    logger.info(s"Train Correct predictions%: ${result.correctPercent}")
  }

  def debugFalsePrediction(prediction: INDArray, y: INDArray): Unit = {
    logger.debug("\r\nPredicted[{}] as [{}]\r\nRealValue[{}] is [{}]", prediction, LabelConverter.vectorToLabel(prediction),
      y, LabelConverter.vectorToLabel(y))
  }
}