package model.nn

import breeze.linalg.DenseMatrix

class OutputLayer(override val thetas: DenseMatrix[Double]) extends SinkLayer {

  override val units = thetas.rows
  override val inputs = thetas.cols - 1

}