package main

import javax.swing.{JFrame, SwingUtilities, WindowConstants}

import com.typesafe.scalalogging.StrictLogging
import model.data.SampleSet
import model.nn.{HiddenLayer, InputLayer, OutputLayer}
import org.jfree.chart.ChartPanel
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import util.RandomInitializier
import util.data.{DataSampler, LabelConverter, MatlabImporter}
import util.optimizers.{AdaDeltaOptimizer, GradientDescendOptimizer, MomentumOptimizer, NesterovAcceleratedOptimizer}
import util.plot.PlotCost

/**
  *
  */
object GradientMain extends StrictLogging {

  def main(args: Array[String]): Unit = {
    val map = MatlabImporter("src/main/resources/test.mat")
    val x = map("X")
    val y = map("y")
    val yMappedCols = Nd4j.zeros(10, y.rows())
    for (i <- 0 until y.rows) {
      val yVal: Int = y(i, 0).intValue()
      //      val mappedY = LabelConverter.labelToVector(yVal, 10)
      LabelConverter.labelToVector(yMappedCols(->, i), yVal, 10)
      //      yMappedCols(->, i) = mappedY
    }

    //    val yReshaped = yMappedCols.reshape('f', 10, y.rows())

    val inputsSource = x.columns()
    val hiddenLayer1Size = 35
    val hiddenLayer2Size = 35
    val labels = 10
    val iterations = 200
    val lambda = 4
    val learnRate = 2

    val theta1 = RandomInitializier.initialize(hiddenLayer1Size, inputsSource, 1)
    val theta2 = RandomInitializier.initialize(hiddenLayer2Size, hiddenLayer1Size, 1)
    val theta3 = RandomInitializier.initialize(labels, hiddenLayer2Size, 1)
    //    val theta3 = RandomInitializier.initialize(labels, hiddenLayer1Size, 1)

    val inputLayer = new InputLayer(inputsSource)
    val hiddenLayer1 = new HiddenLayer(theta1)
    val hiddenLayer2 = new HiddenLayer(theta2)
    val outputLayer = new OutputLayer(theta3)
    inputLayer.connectTo(hiddenLayer1)
    //    hiddenLayer1.connectTo(outputLayer)
    hiddenLayer1.connectTo(hiddenLayer2)
    hiddenLayer2.connectTo(outputLayer)

    val dataset = DataSampler.createSampleSet(x, yMappedCols)

    val optimizers: Seq[(String, (INDArray, INDArray, InputLayer, Int, Double, Double) => Seq[Double])] =
      Seq(
        ("Gradient Descent", GradientDescendOptimizer.minimize),
        ("Momentum", MomentumOptimizer.minimize(_, _, _, _, _, _)),
        ("Nesterov", NesterovAcceleratedOptimizer.minimize(_, _, _, _, _, _)),
        ("Ada Delta", AdaDeltaOptimizer.minimize)
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

    val runnable: Runnable = () => {
      val panel = new ChartPanel(PlotCost.plot(nameAndCosts))
      val frame = new JFrame()
      val factor = 100
      frame.setSize(16 * factor, 9 * factor)
      frame.setLocationRelativeTo(null)
      frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
      frame.setContentPane(panel)
      frame.setVisible(true)
    }
    SwingUtilities.invokeLater(runnable)
  }


  private def runOnCvSet(inputLayer: InputLayer, dataset: SampleSet) = {
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