package model.nn

import model.nn.Layers.SourceLayer

class InputLayer(size: Int) extends SourceLayer {

  val inputs: Int = size
  val units: Int = size

}