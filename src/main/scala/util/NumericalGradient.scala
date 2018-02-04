package util

import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object NumericalGradient extends StrictLogging {

  def approximateGradients(costFunction: (() => Double), thetas: Seq[INDArray]): Seq[INDArray] = {
    val eps = 0.0001D
    val sum = thetas.map(_.length()).sum
    var count = 0
    val tenPercent = sum / 100 * 10
    thetas.map(theta => {
      val rows = theta.rows()
      val columns = theta.columns
      val gradients = Nd4j.zeros(rows, columns)
      for {
        row <- 0 until rows
        col <- 0 until columns
      } yield {
        val curTheta = theta(row, col)

        theta(row, col) = curTheta + eps
        val c1 = costFunction()

        theta(row, col) = curTheta - eps
        val c2 = costFunction()

        theta(row, col) = curTheta
        logger.debug("c1[{}] c2[{}]", c1, c2)

        gradients(row, col) = (c1 - c2) / (2 * eps)
        count += 1
        if (count % tenPercent == 0) {
          logger.info("Gradient approximation {}% done", count / tenPercent * 10)
        }
      }

      gradients
    })

  }

}
