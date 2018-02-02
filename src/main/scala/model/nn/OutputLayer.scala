package model.nn

import model.nn.Layers.SinkLayer
import org.nd4j.linalg.api.ndarray.INDArray

class OutputLayer(override var thetas: INDArray) extends SinkLayer {

  override val units = thetas.rows
  override val inputs = thetas.columns - 1

}