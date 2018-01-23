package model.nn

import breeze.linalg.DenseMatrix

class HiddenLayer(private val thetas: DenseMatrix[Double]) extends Layer {

  val units = thetas.rows
  val inputs = thetas.cols - 1

}