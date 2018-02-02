package model.nn

import model.nn.Layers.ConnectableLayer
import org.nd4j.linalg.api.ndarray.INDArray

class HiddenLayer(override var thetas: INDArray) extends ConnectableLayer {

  override val units = thetas.rows
  override val inputs = thetas.columns - 1

}