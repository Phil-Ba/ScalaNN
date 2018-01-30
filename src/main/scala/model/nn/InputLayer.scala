package model.nn

import model.nn.Layers.SourceLayer

class InputLayer(size: Int) extends SourceLayer {

  val units: Int = size

}