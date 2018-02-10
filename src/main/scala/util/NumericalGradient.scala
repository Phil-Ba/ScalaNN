package util

import com.typesafe.scalalogging.StrictLogging
import model.nn.InputLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object NumericalGradient extends StrictLogging {


  //  def approximateGradients(costFunction: (() => Double)): Seq[INDArray] = {
  def approximateGradients(costFunction: () => Double, inputLayer: InputLayer, shapes: Seq[Array[Int]]): Seq[INDArray] = {
    val eps = 0.0001D
    val epsTimesTwo = 2D * eps
    val thetas = shapes.map(shape => Nd4j.zeros(shape(0), shape(1)))
    val sum = thetas.map(_.length()).sum
    var count = 0
    val tenPercent = math.ceil(sum / 100D * 10)
    thetas.map(theta => {
      val rows = theta.rows()
      val columns = theta.columns
      val gradients = Nd4j.zeros(rows, columns)
      for {
        row <- 0 until rows
        col <- 0 until columns
      } yield {
        theta(row, col) = -eps
        inputLayer.updateWithGradients(thetas)
        val c1 = costFunction()

        theta(row, col) = epsTimesTwo
        inputLayer.updateWithGradients(thetas)
        val c2 = costFunction()

        theta(row, col) = -eps
        inputLayer.updateWithGradients(thetas)
        logger.debug("c1[{}] c2[{}]", c1, c2)

        gradients(row, col) = (c1 - c2) / epsTimesTwo
        count += 1
        if (count % tenPercent == 0) {
          logger.info("Gradient approximation {}% done", count / tenPercent * 10)
        }
      }

      gradients
    })

  }

}
