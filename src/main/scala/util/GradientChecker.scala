package util

import model.nn.{HiddenLayer, InputLayer, OutputLayer}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

/**
  *
  */
object GradientChecker {

  def check(x: INDArray, y: INDArray) = {
    val inputsSource = 3
    val hiddenLayerSize = 5
    val labels = 3
    val testDataAmount = 1

    val x = RandomInitializier.initialize(testDataAmount, inputsSource - 1, 0.5)
    val y = Nd4j.zeros(labels, testDataAmount)
    for (sample <- 0 until testDataAmount) {
      y(sample % (labels - 1), sample) = 1
    }

    //hls X inpS+1
    val theta1 = RandomInitializier.initialize(hiddenLayerSize, inputsSource, 0.5)
    //10 X hls+1
    val theta2 = RandomInitializier.initialize(labels, hiddenLayerSize, 0.5)

    val inputLayer = new InputLayer(inputsSource)
    val hiddenLayer = new HiddenLayer(theta1)
    val outputLayer = new OutputLayer(theta2)
    inputLayer.connectTo(hiddenLayer)
    hiddenLayer.connectTo(outputLayer)

    val (_, gradients, _) = inputLayer.activateWithGradients(x, y)
    val costFunction = () => CostFunction.cost(() => inputLayer.activate(x), y)
    val gradientsApprox = NumericalGradient.approximateGradients(costFunction, Seq(theta1, theta2))


    val gradT1 = gradients.head
    val gradT1Approx = gradientsApprox.head
    val gradT2 = gradients.last
    val gradT2Approx = gradientsApprox.last
    println(gradT1.shapeInfoToString())
    println(gradT1Approx.shapeInfoToString())
    println(gradT2.shapeInfoToString())
    println(gradT2Approx.shapeInfoToString())

    val diffT1 = Transforms.abs(gradT1 - gradT1Approx).sumNumber().doubleValue()
    val diffT2 = Transforms.abs(gradT2 - gradT2Approx).sumNumber().doubleValue()

    println(s"diffT1[{$diffT1}]")
    println(s"diffT1[{${diffT1 / gradT1.length()}]")
    println(s"diffT2[{$diffT2}]")
    println(s"diffT1[{${diffT2 / gradT2.length()}]")

    println("-----t1------")
    println(Nd4j.hstack(gradT1.reshape(gradT1.length(), 1), gradT1Approx.reshape(gradT1.length(), 1)))
    println("-----t2------")
    println(Nd4j.hstack(gradT2.reshape(gradT2.length(), 1), gradT2Approx.reshape(gradT2.length(), 1)))
  }

}
