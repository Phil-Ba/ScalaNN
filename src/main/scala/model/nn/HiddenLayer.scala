package model.nn

import breeze.linalg.DenseMatrix

class HiddenLayer(override val thetas: DenseMatrix[Double]) extends ConnectableLayer {

  override val units = thetas.rows
  override val inputs = thetas.cols - 1

}