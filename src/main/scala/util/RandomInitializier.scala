package util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object RandomInitializier {

  def initialize(layerSize: Int, intputs: Int, epsilon: Double = 5): INDArray = {

    Nd4j.rand(layerSize, intputs + 1) * (2 * epsilon) - epsilon

  }

}
