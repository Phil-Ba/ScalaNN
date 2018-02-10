package util

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  *
  */
object GradientDescender extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, thetas: Seq[INDArray]) = {

    for (i <- 1 to 100) {
      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer)
      val cost = CostFunction.cost(yCalc, y)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      thetas.zip(gradients)
        .foreach(t => {
          val (curTheta, curGradient) = t
          curTheta -= curGradient * 0.4
        })
    }

  }


}
