package model.nn

import breeze.linalg.DenseMatrix

class HiddenLayer(override val thetas: DenseMatrix[Double]) extends ConnectableLayer {

  val units = thetas.rows
  val inputs = thetas.cols - 1

}