package util

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  *
  */
object GradientDescender extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, lambda: Double = 0, learnRate: Double = 1D) = {
    for (i <- 1 to 30) {
      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)
      val cost = CostFunction.cost(yCalc, y, inputLayer.getNNThetas, lambda)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      gradients.foreach(_ *= learnRate)
      inputLayer.updateWithGradients(gradients)
    }

  }

}
