package util.optimizers

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.{CostFunction, NNRunner}

/**
  *
  */
object AdaDeltaOptimizer extends StrictLogging {

  def minimize(x: INDArray, y: INDArray, inputLayer: InputLayer, iterations: Int = 100, lambda: Double = 0, learnRate: Double = 1D) = {
    //decay
    val gamma = 0.9
    val epsilon = 0.0000001
    val msg: Seq[INDArray] = inputLayer.getNNThetas.map {
      Nd4j.onesLike(_)
    }
    val msdx: Seq[INDArray] = inputLayer.getNNThetas.map {
      Nd4j.onesLike(_)
    }

    val costs = (1 to iterations).map { i =>
      val (yCalc, gradients) = NNRunner.runWithData(x, y, inputLayer, lambda)
      //E[g^2]_t = gamma * E[g^2]_{t−1} + (1-gamma)*g^2_t
      msg.zip(gradients).foreach { msgG =>
        val (msg, gradient) = msgG
        msg.muli(gamma).addi(Transforms.pow(gradient, 2, false).muli(1 - gamma))
      }

      //Calculate update:
      //dX = - g * RMS[delta x]_{t-1} / RMS[g]_t
      //Note: negative is applied in the DL4J step function: params -= update rather than params += update
      val rmsdx_t1 = msdx.map { msdx =>
        Transforms.sqrt(msdx.add(epsilon), false)
      }
      val rmsg_t = msg.map { msg =>
        Transforms.sqrt(msg.add(epsilon), false)
      }
      val update = (gradients, rmsdx_t1, rmsg_t).zipped
        .map { (grad, dx, gt) =>
          grad.mul(dx.divi(gt))
        }
      //        .map { gradDxGt =>
      //          val (grad, dx, rms) = gradDxGt
      //          ""
      //        }
      //      val update = gradients.zipAll(rmsdx_t1,rmsg_t).map { gradRms =>
      //        val(grad,rmsdx_t1)=gradRms
      //        grad.mul(rmsdx_t1.divi(rmsg_t))
      //      }

      msdx.zip(update)
        //      Accumulate gradients: E[delta x^2]_t = rho * E[delta x^2]_{t-1} + (1-rho)* (delta x_t)^2
        .foreach { msdxUp =>
        val (msdx, update) = msdxUp
        msdx.muli(gamma).addi(Transforms.pow(update, 2, true).muli(1 - gamma))
      }
      //      msdx.foreach(_.muli(gamma).addi(update.mul(update).muli(1 - rho))

      val cost = CostFunction.cost(yCalc, y, inputLayer.getNNThetas, lambda)
      logger.info("Iteration[{}] cost:[{}]", i, cost)
      inputLayer.updateWithGradients(update)
      cost
    }
    costs
  }

}
