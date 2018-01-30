package model.nn

import breeze.linalg.DenseMatrix
import model.nn.Layers.ConnectableLayer

class HiddenLayer(override var thetas: DenseMatrix[Double]) extends ConnectableLayer {

  override val units = thetas.rows
  override val inputs = thetas.cols - 1

}