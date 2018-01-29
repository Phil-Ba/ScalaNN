package model.nn

import breeze.linalg.DenseMatrix

class InputLayer(private val data: DenseMatrix[Double]) extends SourceLayer {

  val units: Int = data.cols

}