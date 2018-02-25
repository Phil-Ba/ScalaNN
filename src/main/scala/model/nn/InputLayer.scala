package model.nn

import model.nn.Layers.{ConnectableLayer, SourceLayer}

class InputLayer(size: Int) extends SourceLayer {

  val inputs: Int = size
  val units: Int = size

  def copyNetwork: InputLayer = {
    val nnThetas = getNNThetas
    val nnCopy = nnThetas.zipWithIndex
      .map { thetaIdx =>
        val (theta, idx) = thetaIdx
        idx match {
          case _ if idx == nnThetas.size - 1 => new OutputLayer(theta)
          case _ => new HiddenLayer(theta)
        }
      }
    val inputLayerCopy = new InputLayer(size)
    nnCopy.foldLeft(inputLayerCopy.asInstanceOf[ConnectableLayer]) { (l1, l2) =>
      l1.nextLayer match {
        case Some(l: ConnectableLayer) =>
          l.connectTo(l2)
          l
        case None =>
          l1.connectTo(l2)
          l1
      }
    }
    inputLayerCopy
  }

}