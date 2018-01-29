package model.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import util.Sigmoid

trait Layer {
  type Result = DenseVector[Double]
  type Delta = DenseVector[Double]
  type Gradients = DenseMatrix[Double]
  type Thetas = DenseMatrix[Double]


  var thetas: Thetas
  val units: Int
  val inputs: Int

  protected def activate(x: DenseVector[Double]): Result

  protected def activateWithGradients(x: DenseVector[Double],
                                      y: DenseVector[Double]): (Result, Seq[Gradients], Delta)

}

trait SourceLayer extends Layer {

  override val inputs = 0
  override val thetas = DenseMatrix.zeros(0, 0)

  var nextLayer: Option[Layer]

  def connectTo(nextLayer: Layer): Unit = {
    require(nextLayer.inputs == this.units)
    this.nextLayer = Option(nextLayer)
  }

  override def activate(x: DenseVector[Double]): Result = {
    require(nextLayer.isDefined)
    require(x.length == this.units)
    nextLayer.get.activate(x)
  }

  override def activateWithGradients(x: DenseVector[Double], y: DenseVector[Double]) = {
    require(nextLayer.isDefined)
    require(x.length == this.units)
    require(x.length == y.length)

    nextLayer.get.activateWithGradients(x, y)
  }

}

trait SinkLayer extends Layer {

}

trait ConnectableLayer extends Layer {

  var nextLayer: Option[Layer]

  private def doActivation(x: DenseVector[Double]) = {
    val a = thetas * x
    val z = Sigmoid.sigmoid(a)
    (a, z)
  }


  override protected def activate(x: DenseVector[Double]): Result = {
    require(nextLayer.isDefined)
    require(x.length == this.units)

    val (activation, z) = doActivation(x)
    nextLayer.get.activate(z)
  }

  override protected def activateWithGradients(x: DenseVector[Double],
                                               y: DenseVector[Double]): (Result, Seq[Gradients], Delta) = {
    require(nextLayer.isDefined)
    require(x.length == this.units)

    val (activation, z) = doActivation(x)
    nextLayer.get.activateWithGradients(z, y)
  }

  def connectTo(nextLayer: Layer): Unit = {
    require(nextLayer.inputs == this.units)
    this.nextLayer = Option(nextLayer)
  }

}
