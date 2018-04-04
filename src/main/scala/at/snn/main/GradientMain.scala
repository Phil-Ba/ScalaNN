package at.snn.main

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.data.{DataSampler, LabelConverter, MatlabImporter}
import at.snn.util.optimizers.{GradientDescendOptimizer, MomentumOptimizer, NesterovAcceleratedOptimizer}
import at.snn.util.plot.{ChartRenderer, PlotCost}
import at.snn.util.{NNBuilder, NNRunner}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  *
  */
object GradientMain extends StrictLogging {

  def main(args: Array[String]): Unit = {
    Nd4j.setDataType(Type.DOUBLE)

    val rawData = MatlabImporter("src/main/resources/test.mat")
    val (x, yMappedCols) = rawData.getData

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