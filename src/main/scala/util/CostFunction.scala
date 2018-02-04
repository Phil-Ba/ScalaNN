package util

import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
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

  //  J += (1/m) * (-yVec'*log(a3) - (1-yVec)'*log(1-a3));
  def cost(calcY: INDArray, y: INDArray): Double = {
    require(calcY.shape() sameElements y.shape(), s"calcy(${calcY.shape().mkString("x")}) should be same as y(${y.shape().mkString("x")})")
    val m = y.columns()
    val avg = 1D / m
    var cost = 0D
    for (i <- 0 until m) {
      val yT = y(->, i).T
      val curCalcY = calcY(->, i)
      val logCalcY = Transforms.log(curCalcY)
      val logOneMinusCalcY = Transforms.log(curCalcY.rsub(1D))

      val c = (yT.neg() dot logCalcY) - (yT.rsub(1D) dot logOneMinusCalcY)
      require(c.shape() sameElements Array(1, 1))
      val curCost = c(0, 0) * avg
      logger.debug("cost for y[{}] and yCalc[{}] = {}", yT, curCalcY, curCost)
      cost += curCost
    }

    cost
  }

}
