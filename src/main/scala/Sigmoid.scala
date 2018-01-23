import breeze.linalg.DenseMatrix

/**
  *
  */
object Sigmoid {

  def sigmoid(m: DenseMatrix[Double]): DenseMatrix[Double] = {

    val mTemp = DenseMatrix.fill(m.rows, m.cols)(breeze.numerics.constants.E)
    val value = ((mTemp ^:^ -m) + 1.0) ^:^ -1.0
    value
  }
}
