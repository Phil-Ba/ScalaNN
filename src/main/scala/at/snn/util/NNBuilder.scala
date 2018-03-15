package at.snn.util

import at.snn.model.nn.Layers.ConnectableLayer
import at.snn.model.nn.{HiddenLayer, InputLayer, OutputLayer}

/**
  *
  */
object NNBuilder {

  def buildNetwork(inputs: Int, labels: Int, layerSizes: Int*): InputLayer = {
    val inputLayer = new InputLayer(inputs)

    val lastLayer = layerSizes.foldLeft(inputLayer.asInstanceOf[ConnectableLayer]) { case (previousLayer, layerSize) =>
      val layer = new HiddenLayer(RandomInitializier.initialize(layerSize, previousLayer.units))
      previousLayer.connectTo(layer)
      layer
    }

    val outputLayer = new OutputLayer(RandomInitializier.initialize(labels, lastLayer.units))
    lastLayer.connectTo(outputLayer)

    inputLayer
  }
}