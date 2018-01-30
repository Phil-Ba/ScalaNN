package model.nn

import breeze.linalg.DenseMatrix
import model.nn.Layers.SinkLayer

class OutputLayer(override var thetas: DenseMatrix[Double]) extends SinkLayer {

  override val units = thetas.rows
  override val inputs = thetas.cols - 1

}