package at.snn.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object RandomInitializier {

  def initialize(layerSize: Int, inputs: Int, epsilon: Double = 1): INDArray = {
    Nd4j.rand(layerSize, inputs + 1) * (2 * epsilon) - epsilon
  }

}