package at.snn.util

import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

/**
  *
  */
object CostFunction extends StrictLogging {

  def cost(fun: INDArray => INDArray, x: INDArray, y: INDArray): Double = {
    cost(fun(x), y)
  }

  def cost(fun: () => INDArray, y: INDArray): Double = {
    cost(fun(), y)
  }

  def cost(calcY: INDArray, y: INDArray, thetas: Seq[INDArray], lambda: Double = 0): Double = {
    val costJ = cost(calcY, y)
    if (lambda == 0) {
      costJ
    } else {
      val m = y.columns()
      val penaltyFactor = lambda / (2 * m)
      val thetaCost = thetas
        .map { theta => Transforms.pow(theta(->, 1 until theta.columns()), 2, true).sumNumber().doubleValue() }
        .sum * penaltyFactor
      costJ + thetaCost
    }
  }

  //  J += (1/m) * (-yVec'*log(a3) - (1-yVec)'*log(1-a3));
  def cost(calcY: INDArray, y: INDArray): Double = {
    require(calcY.shape() sameElements y.shape(), s"calcy(${calcY.shape().mkString("x")}) should be same as y(${y.shape().mkString("x")})")
    val m = y.columns()
    val logCalcY = Transforms.log(calcY)
    val logOneMinusCalcY = Transforms.log(calcY.rsub(1D))
    val c = (y.neg() * logCalcY) - (y.rsub(1D) * logOneMinusCalcY)
    c.sumNumber().doubleValue() / m
  }

}
