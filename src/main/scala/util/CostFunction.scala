package util

import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

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
        .map { theta => Transforms.pow(theta(->, 2 until theta.columns()), 2).sumNumber().doubleValue() }
        .sum * penaltyFactor
      costJ + thetaCost
    }
  }

  //  J += (1/m) * (-yVec'*log(a3) - (1-yVec)'*log(1-a3));
  def cost(calcY: INDArray, y: INDArray): Double = {
    require(calcY.shape() sameElements y.shape(), s"calcy(${calcY.shape().mkString("x")}) should be same as y(${y.shape().mkString("x")})")
    val m = y.columns()
    val avg = 1D / m
    val costVec = Nd4j.zeros(m)
    for (i <- 0 until m) {
      val yT = y(->, i).T
      val curCalcY = calcY(->, i)
      val logCalcY = Transforms.log(curCalcY)
      val logOneMinusCalcY = Transforms.log(curCalcY.rsub(1D))

      val c = (yT.neg() dot logCalcY) - (yT.rsub(1D) dot logOneMinusCalcY)
      require(c.isScalar)
      val curCost = c(0, 0)
      logger.debug("cost for y[{}] and yCalc[{}] = {}", yT, curCalcY, curCost)
      costVec(i) = curCost
    }

    costVec.sumNumber().doubleValue() * avg
  }

}
