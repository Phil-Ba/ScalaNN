package at.snn.main

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.data.{DataSampler, LabelConverter, MatlabImporter}
import at.snn.util.optimizers.{GradientDescendOptimizer, MomentumOptimizer, NesterovAcceleratedOptimizer, ParamOptimizer}
import at.snn.util.plot.{ChartRenderer, PlotCost}
import at.snn.util.{NNBuilder, NNRunner}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  *
  */
object OptimizedParametersMain extends StrictLogging {

  def main(args: Array[String]): Unit = {
    Nd4j.setDataType(Type.DOUBLE)

    val rawData = MatlabImporter("src/main/resources/test.mat")
    val (x, yMappedCols) = rawData.getData

    val inputsSource = x.columns()
    val labels = 10
    val iterations = 200
    val learnRate = 1
    val deltaValues = Seq(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    val lambdaValues = Seq(0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6)
    val inputLayer = NNBuilder.buildNetwork(inputsSource, labels, 35, 35, 35, 35)

    val dataset = DataSampler.createSampleSet(x, yMappedCols)

    val gradOptimizer: (InputLayer, Double) => Seq[Double] = GradientDescendOptimizer
      .minimize(dataset.trainingSet, dataset.trainingResultSet, _, 50, _, learnRate)
    val gradLambdaParam = ParamOptimizer
      .optimizeByCost(dataset, inputLayer, gradOptimizer, lambdaValues)
      ._1

    val momentumOptimizer: (InputLayer, Double, Double) => Seq[Double] = MomentumOptimizer
      .minimize(dataset.trainingSet, dataset.trainingResultSet, _, 50, _, learnRate, _)
    val (momLambdaParam, momDeltaParam, _) = ParamOptimizer
      .optimizeByCost(dataset, inputLayer, momentumOptimizer, lambdaValues, deltaValues)

    val nesterovOptimizer: (InputLayer, Double, Double) => Seq[Double] = NesterovAcceleratedOptimizer
      .minimize(dataset.trainingSet, dataset.trainingResultSet, _, 50, _, learnRate, _)
    val (nesterovLambdaParam, nesterovDeltaParam, _) = ParamOptimizer
      .optimizeByCost(dataset, inputLayer, nesterovOptimizer, lambdaValues, deltaValues)

    val optimizers: Seq[(String, (INDArray, INDArray, InputLayer, Int, Double) => Seq[Double])] =
      Seq(
        (s"Gradient Descent($gradLambdaParam)", GradientDescendOptimizer.minimize(_, _, _, _, gradLambdaParam, _)),
        (s"Momentum($momLambdaParam, momDeltaParam)", MomentumOptimizer.minimize(_, _, _, _, momLambdaParam, _, momDeltaParam)),
        (s"Nesterov($nesterovLambdaParam, $nesterovDeltaParam)", NesterovAcceleratedOptimizer
          .minimize(_, _, _, _, nesterovLambdaParam, _, nesterovDeltaParam))
      )

    val nameAndCosts = optimizers
      .map({ case (name, optimizer) =>
        val network = inputLayer.copyNetwork
        val costs = optimizer(dataset.trainingSet, dataset.trainingResultSet, network, iterations, learnRate)
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