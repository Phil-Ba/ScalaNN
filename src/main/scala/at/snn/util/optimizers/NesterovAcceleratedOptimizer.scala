package at.snn.util.optimizers

import at.snn.model.nn.InputLayer
import at.snn.util.{CostFunction, NNRunner}
import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits.RichINDArray

/**
  * Implementation according to http://cs231n.github.io/neural-networks-3/#sgd
  */
object NesterovAcceleratedOptimizer extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, iterations: Int = 100
               , lambda: Double = 0, learnRate: Double = 1D, momentum: Double = 0.9): Seq[Double] = {

    var vtMinusOne = inputLayer.getNNThetas
      .map(Nd4j.zerosLike(_))
    var vt = inputLayer.getNNThetas
      .map(Nd4j.zerosLike(_))

    val costs = (1 to iterations).map { i =>
      val update = vtMinusOne.zip(vt)
        .map { case (vtMinusOneLayer, vtLayer) =>
          vtMinusOneLayer.muli(momentum).subi(vtLayer * (1 + momentum))
        }
      inputLayer.updateWithGradients(update)

      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)
      val cost = CostFunction.cost(yCalc, y, inputLayer.getNNThetas, lambda)
      logger.info("Iteration[{}] cost:[{}]", i, cost)

      vtMinusOne = vt

      if (learnRate != 1) {
        gradients.foreach(_ *= learnRate)
      }

      vt = vt.zip(gradients).map { case (vtLayer, grads) =>
        (vtLayer * momentum) -= grads
      }

      cost
    }
    costs
  }

}
