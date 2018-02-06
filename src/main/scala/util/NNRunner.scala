package util

import model.nn.InputLayer
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
  def runWithData(x: INDArray, y: INDArray, inputLayer: InputLayer): (INDArray, Seq[inputLayer.Gradients]) = {
    require(x.rows() == y.columns(), s"X.rows[{${x.rows()}}] must be the same as Y.columns[{${y.columns()}}]")

    val yCalc = Nd4j.zerosLike(y)
    val m = y.columns()
    val gradients = for {
      i <- 0 until m
      xCur = x(i, ->)
      yCur = y(->, i)
    } yield {
      val (result, gradients, _) = inputLayer.activateWithGradients(xCur, yCur)
      yCalc(->, i) = result
      gradients
    }

    val avg = 1D / m
    val totalGradients = gradients.reduce((gradients1, gradients2) => {
      val sumOfLayerGradients = gradients1.zip(gradients2)
        .map(layerGradients => {
          val (layerGradients1, layerGradients2) = layerGradients
          layerGradients1 + layerGradients2
        })
      sumOfLayerGradients
    }).map(layerGradients => layerGradients * avg)

    (yCalc, totalGradients)
  }

  /**
    * Feeds the input layer with data and accumulates the results
    *
    * @param x          input data mXn, where the columns are features and rows are the samples
    * @param inputLayer layer to feed the data to
    * @return calculated results
    */
  def runWithData(x: INDArray, inputLayer: InputLayer): (INDArray) = {
    val yCalc = for {
      i <- 0 until x.rows()
      xCur = x(i, ->)
    } yield {
      inputLayer.activate(xCur)
    }
    Nd4j.hstack(yCalc: _*)
  }

}
