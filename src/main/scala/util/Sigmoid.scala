package util

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  *
  */
object Sigmoid {

  def sigmoid(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val mTemp = DenseMatrix.fill(m.rows, m.cols)(breeze.numerics.constants.E)
    val value = ((mTemp ^:^ -m) + 1.0) ^:^ -1.0
    value
  }

  def sigmoid(v: DenseVector[Double]): DenseVector[Double] = {
    val vTemp = DenseVector.fill(v.length, breeze.numerics.constants.E)
    val value = ((vTemp ^:^ -v) + 1.0) ^:^ -1.0
    value
  }
}
