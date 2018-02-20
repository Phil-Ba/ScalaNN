package util

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  *
  */
object GradientDescender extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, iterations: Int = 100, lambda: Double = 0, learnRate: Double = 1D) = {

    val gamma = 0.9
    var v = inputLayer.getNNThetas
      .map(Nd4j.zerosLike(_))
    for (i <- 1 to iterations) {

      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)
      val cost = CostFunction.cost(yCalc, y, inputLayer.getNNThetas, lambda)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      //      if (learnRate != 1) {
      //        gradients.foreach(_ *= learnRate)
      //      }

      val vtNew = v.zip(gradients).map { (vtGrad) =>
        val (vt, gradients) = vtGrad
        vt.mul(gamma).subi(gradients.mul(learnRate))
      }
      val update = v.zip(vtNew)
        .map { vVnew =>
          val (v, vtNew) = vVnew
          v.mul(gamma).addi(vtNew.mul(-gamma - 1))
        }
      inputLayer.updateWithGradients(update)
      v = vtNew
    }

  }

}
