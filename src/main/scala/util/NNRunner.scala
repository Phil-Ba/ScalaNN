package util

import model.nn.InputLayer
import model.nn.Layers.Layer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits.{->, _}

/**
  *
  */
object NNRunner {

  /**
    * Feeds the input layer with data and accumulates the results
    *
    * @param x          input data mXn, where the columns are features and rows are the samples
    * @param y          expected output data lXm, where the columns are the samples ans the rows are the classes
    * @param inputLayer layer to feed the data to
    * @return calculated results and accumulated gradients
    */
  def runWithData(x: INDArray, y: INDArray, inputLayer: InputLayer, lambda: Double = 0): (INDArray, Seq[Layer#Gradients]) = {
    require(x.rows() == y.columns(), s"X.rows[{${x.rows()}}] must be the same as Y.columns[{${y.columns()}}]")

    val yCalc = Nd4j.zerosLike(y)
    val m = y.columns()
    val lambdaM = lambda / m
    val avg = 1D / m
    val (result, gradients, _) = inputLayer.activateWithGradients(x, y, 0)
    val totalGradients = gradients.map(layerGradients => layerGradients * avg)
      .zip(inputLayer.getNNThetas)
      .map { gt =>
        val thetasExclBias = gt._2(->, 1 until gt._2.columns())
        gt._1(->, 1 until gt._1.columns()) = gt._1(->, 1 until gt._1.columns()) + (thetasExclBias * lambdaM)
      }
    (result, totalGradients)
  }

  /**
    * Feeds the input layer with data and accumulates the results
    *
    * @param x          input data mXn, where the columns are features and rows are the samples
    * @param inputLayer layer to feed the data to
    * @return calculated results
    */
  def runWithData(x: INDArray, inputLayer: InputLayer): (INDArray) = {
    inputLayer.activate(x)
  }

}
