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
    val hiddenLayer1Size = 25
    val hiddenLayer2Size = 25
    val labels = 10

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
    GradientDescender.minimize(dataset.trainingSet, dataset.trainingResultSet, inputLayer, 200, 4, 2)

    //    val (cvSet, cvResultSet) = DataSampler.sample(dataset.cvSet, dataset.cvResultSet, 10)
    val tSet = dataset.trainingSet
    val tResultSet = dataset.trainingResultSet
    var falseCount = 0
    for (i <- 0 until tSet.rows()) {
      val result = inputLayer.activate(tSet(i, ->))
      val y = tResultSet(->, i)
      val yLabel = LabelConverter.vectorToLabel(y)
      val predictLabel = LabelConverter.vectorToLabel(result)
      if (yLabel != predictLabel) {
        //        println(s"Expected ${yLabel} and got ${predictLabel}")
        falseCount += 1
      }
    }
    println(s"Total predictions: ${tSet.rows()}")
    println(s"Incorrect predictions: ${falseCount}")
    println(s"Correct predictions%: ${100 - (falseCount / tSet.rows().toDouble) * 100D}")
    println("---------------------\r\n")

    val cvSet = dataset.cvSet
    val cvResultSet = dataset.cvResultSet
    falseCount = 0
    for (i <- 0 until cvSet.rows()) {
      val result = inputLayer.activate(cvSet(i, ->))
      val y = cvResultSet(->, i)
      val yLabel = LabelConverter.vectorToLabel(y)
      val predictLabel = LabelConverter.vectorToLabel(result)
      if (yLabel != predictLabel) {
        //        println(s"Expected ${yLabel} and got ${predictLabel}")
        falseCount += 1
      }
    }
    println(s"CV Total predictions: ${cvSet.rows()}")
    println(s"CV Incorrect predictions: ${falseCount}")
    println(s"CV Correct predictions%: ${100 - (falseCount / cvSet.rows().toDouble) * 100D}")
  }

}