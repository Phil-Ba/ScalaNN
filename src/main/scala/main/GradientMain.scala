package main

import model.nn.{HiddenLayer, InputLayer, OutputLayer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import util.{GradientDescender, MatlabImporter, RandomInitializier}

/**
  *
  */
object GradientMain {

  def main(args: Array[String]): Unit = {
    val map = MatlabImporter("src/main/resources/test.mat")
    val x = map("X")
    val y = map("y")
    val yMappedCols = Nd4j.zeros(1, y.rows() * 10)
    for (i <- 0 until y.rows) {
      val yVal: Int = y(i, 0).intValue()
      val updIdx = (yVal % 10) + 10 * i
      yMappedCols(0, updIdx) = 1
    }

    val yReshaped = yMappedCols.reshape('f', 10, y.rows())

    val inputsSource = x.columns()
    val hiddenLayer1Size = 4
    val hiddenLayer2Size = 5
    val labels = 10

    val theta1 = RandomInitializier.initialize(hiddenLayer1Size, inputsSource, 0.5)
    val theta2 = RandomInitializier.initialize(hiddenLayer2Size, hiddenLayer1Size, 0.5)
    val theta3 = RandomInitializier.initialize(labels, hiddenLayer2Size, 0.5)

    val inputLayer = new InputLayer(inputsSource)
    val hiddenLayer1 = new HiddenLayer(theta1)
    val hiddenLayer2 = new HiddenLayer(theta2)
    val outputLayer = new OutputLayer(theta3)
    inputLayer.connectTo(hiddenLayer1)
    hiddenLayer1.connectTo(hiddenLayer2)
    hiddenLayer2.connectTo(outputLayer)

    GradientDescender.minimize(x, yReshaped, inputLayer)
  }

}