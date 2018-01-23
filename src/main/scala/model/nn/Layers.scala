package model.nn

import breeze.linalg.{DenseMatrix, DenseVector}

trait Layer {
  val units: Int
  val inputs: Int

  def activate(x: DenseVector[Double]): (DenseVector[Double], Seq[DenseMatrix[Double]])

}

trait SourceLayer extends Layer {

  override val inputs = 0

  var nextLayer: Option[Layer]

  def connectTo(nextLayer: Layer): Unit = {
    require(nextLayer.inputs == this.units)
    this.nextLayer = Option(nextLayer)
  }

  def activate(x: DenseVector[Double]) = {
    require(x.length == this.units)
    nextLayer.get.activate(x)
  }

}

trait SinkLayer extends Layer {

  def activate(x: DenseVector[Double]) = {
  }

}

trait ConnectableLayer extends Layer {

  var nextLayer: Option[Layer]

  def connectTo(nextLayer: Layer): Unit = {
    require(nextLayer.inputs == this.units)
    this.nextLayer = Option(nextLayer)
  }

}
