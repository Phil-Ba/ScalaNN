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

  protected def doActivation(x: DenseVector[Double]) = {
    val z = thetas * x
    val a = Sigmoid.sigmoid(z)
    (a, z)
  }

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

  override def activate(x: DenseVector[Double]): Result = {
    require(x.length == this.units)
    val (activation, _) = doActivation(x)

    activation
  }

  override def activateWithGradients(x: DenseVector[Double], y: DenseVector[Double]) = {
    require(x.length == this.units)
    require(x.length == y.length)

    val (activation, z) = doActivation(x)
    val delta = activation -:- z
    (activation, Seq.empty, delta)
  }

}

trait ConnectableLayer extends Layer {

  var nextLayer: Option[Layer]

  override protected def activate(x: DenseVector[Double]): Result = {
    require(nextLayer.isDefined)
    require(x.length == this.units)

    val (activation, _) = doActivation(x)
    nextLayer.get.activate(activation)
  }

  override protected def activateWithGradients(x: DenseVector[Double],
                                               y: DenseVector[Double]): (Result, Seq[Gradients], Delta) = {
    require(nextLayer.isDefined)
    require(x.length == this.units)

    val (activation, z) = doActivation(x)
    val (result, gradients, prevDelta) = nextLayer.get.activateWithGradients(activation, y)

    val curDelta = thetas.t * prevDelta *:* z
    val curGradient = prevDelta * activation.t
    (result, curGradient +: gradients, curDelta)
  }

  def connectTo(nextLayer: Layer): Unit = {
    require(nextLayer.inputs == this.units)
    this.nextLayer = Option(nextLayer)
  }

}