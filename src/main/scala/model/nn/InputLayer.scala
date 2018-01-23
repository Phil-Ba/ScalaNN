package model.nn

import breeze.linalg.DenseMatrix

class InputLayer(private val data: DenseMatrix[Double]) extends Layer {

  val units: Int = data.cols
  val inputs: Int = 0

}