package at.snn.util.optimizers

import at.snn.model.nn.InputLayer
import at.snn.util.{CostFunction, NNRunner}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits.RichINDArray

/**
  *
  */
object GradientDescendOptimizer extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, iterations: Int = 100, lambda: Double = 0,
               learnRate: Double = 1D): Seq[Double] = {

    val costs = (1 to iterations).map { i =>
      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)
      val cost = CostFunction.cost(yCalc, y, inputLayer.getNNThetas, lambda)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      if (learnRate != 1) {
        gradients.foreach(_ *= learnRate)
      }
      inputLayer.updateWithGradients(gradients)
      cost
    }
    costs

  }

}
