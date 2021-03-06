package at.snn.util

import at.snn.model.nn.{HiddenLayer, InputLayer, OutputLayer}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

/**
  *
  */
object GradientChecker extends StrictLogging {

  def check(lambda: Double = 0): Seq[(Double, Double)] = {
    val inputsSource = 3
    val hiddenLayer1Size = 4
    val hiddenLayer2Size = 5
    val labels = 3
    val testDataAmount = 100

    val x = RandomInitializier.initialize(testDataAmount, inputsSource - 1, 50)
    val y = Nd4j.zeros(labels, testDataAmount)
    for (sample <- 0 until testDataAmount) {
      y(sample % labels, sample) = 1
    }

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

    val costFunction = () => CostFunction.cost(NNRunner.runWithData(x, inputLayer), y, inputLayer.getNNThetas, lambda)
    val gradientsApprox = NumericalGradient
      .approximateGradients(costFunction, inputLayer, Seq(theta1.shape(), theta2.shape(), theta3.shape()))

    val (_, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)

    val gradT1 = gradients(0)
    val gradT1Approx = gradientsApprox(0)
    val gradT2 = gradients(1)
    val gradT2Approx = gradientsApprox(1)
    val gradT3 = gradients(2)
    val gradT3Approx = gradientsApprox(2)

    val diffT1 = Transforms.abs(gradT1 - gradT1Approx).sumNumber().doubleValue()
    val diffT1Avg = diffT1 / gradT1.length()
    val diffT2 = Transforms.abs(gradT2 - gradT2Approx).sumNumber().doubleValue()
    val diffT2Avg = diffT2 / gradT2.length()
    val diffT3 = Transforms.abs(gradT3 - gradT3Approx).sumNumber().doubleValue()
    val diffT3Avg = diffT3 / gradT3.length()

    logger.debug(s"diffT1 abs[{$diffT1}]")
    logger.debug(s"diffT1 avg[{$diffT1Avg]")
    logger.debug(s"diffT2 abs[{$diffT2}]")
    logger.debug(s"diffT2 avg[{$diffT2Avg]")
    logger.debug(s"diffT3 abs[{$diffT3}]")
    logger.debug(s"diffT3 avg[{$diffT3Avg]")

    logger.debug("-----t1------")
    logger.debug("Unrolled theta1:\r\n{}", Nd4j.hstack(gradT1.reshape(gradT1.length(), 1), gradT1Approx.reshape(gradT1.length(), 1)))
    logger.debug("-----t2------")
    logger.debug("Unrolled theta2:\r\n{}", Nd4j.hstack(gradT2.reshape(gradT2.length(), 1), gradT2Approx.reshape(gradT2.length(), 1)))
    logger.debug("-----t3------")
    logger.debug("Unrolled theta3:\r\n{}", Nd4j.hstack(gradT3.reshape(gradT3.length(), 1), gradT3Approx.reshape(gradT3.length(), 1)))

    Seq((diffT1, diffT2Avg),
      (diffT2, diffT2Avg),
      (diffT2, diffT2Avg)
    )
  }

}
