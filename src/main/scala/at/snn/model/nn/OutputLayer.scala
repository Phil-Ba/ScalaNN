package at.snn.model.nn

import at.snn.model.nn.Layers.SinkLayer
import org.nd4j.linalg.api.ndarray.INDArray

class OutputLayer(thetasInit: INDArray) extends SinkLayer {

  protected[nn] val thetas: Thetas = thetasInit.dup()

  override val units = thetas.rows
  override val inputs = thetas.columns - 1

}