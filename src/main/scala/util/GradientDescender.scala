package util

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  *
  */
object GradientDescender extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer) = {

    for (i <- 1 to 30) {
      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer)
      val cost = CostFunction.cost(yCalc, y)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      gradients.foreach(_ *= 0.4)
      inputLayer.updateWithGradients(gradients)
    }

  }


}
