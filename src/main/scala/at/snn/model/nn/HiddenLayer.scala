package at.snn.model.nn

import at.snn.model.nn.Layers.ConnectableLayer
import org.nd4j.linalg.api.ndarray.INDArray

class HiddenLayer(thetasInit: INDArray) extends ConnectableLayer {

  protected[nn] val thetas: Thetas = thetasInit.dup()

  override val units = thetas.rows
  override val inputs = thetas.columns - 1
}