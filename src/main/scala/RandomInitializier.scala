import breeze.linalg.DenseMatrix

/**
  *
  */
object RandomInitializier {

  def initialize(layerSize: Int, intputs: Int, epsilon: Double = 5): DenseMatrix[Double] = {

    DenseMatrix.rand(layerSize, intputs + 1) *:* (2 * epsilon) - epsilon

  }

}
