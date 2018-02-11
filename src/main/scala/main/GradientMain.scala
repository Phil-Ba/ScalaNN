package main

import model.nn.{HiddenLayer, InputLayer, OutputLayer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import util.data.{DataSampler, LabelConverter, MatlabImporter}
import util.{GradientDescender, RandomInitializier}

/**
  *
  */
object GradientMain {

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
    val hiddenLayer1Size = 15
    val hiddenLayer2Size = 15
    val labels = 10

    val theta1 = RandomInitializier.initialize(hiddenLayer1Size, inputsSource, 1)
    val theta2 = RandomInitializier.initialize(hiddenLayer2Size, hiddenLayer1Size, 1)
    val theta3 = RandomInitializier.initialize(labels, hiddenLayer2Size, 1)

    val inputLayer = new InputLayer(inputsSource)
    val hiddenLayer1 = new HiddenLayer(theta1)
    val hiddenLayer2 = new HiddenLayer(theta2)
    val outputLayer = new OutputLayer(theta3)
    inputLayer.connectTo(hiddenLayer1)
    hiddenLayer1.connectTo(hiddenLayer2)
    hiddenLayer2.connectTo(outputLayer)

    val dataset = DataSampler.createSampleSet(x, yMappedCols)
    GradientDescender.minimize(dataset.trainingSet, dataset.trainingResultSet, inputLayer, 100, 2, 2.5)

    val (cvSet, cvResultSet) = DataSampler.sample(dataset.cvSet, dataset.cvResultSet, 10)
    for (i <- 0 until cvSet.rows()) {
      val result = inputLayer.activate(cvSet(i, ->))
      val y = cvResultSet(->, i)

      println(s"Expected ${LabelConverter.vectorToLabel(y)} and got ${LabelConverter.vectorToLabel(result)}")
    }
  }

}