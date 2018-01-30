package model.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import util.Sigmoid

object Layers {

  trait Layer {
    type Result = DenseVector[Double]
    type Delta = DenseVector[Double]
    type Activation = DenseVector[Double]
    type Z = DenseVector[Double]
    type Gradients = DenseMatrix[Double]
    type Thetas = DenseMatrix[Double]

    protected var thetas: Thetas
    val units: Int
    val inputs: Int

    protected def validateXInput(x: DenseVector[Double]) = {
      require(x.length == this.inputs, s"x.length(${x.length}) | this.inputs($inputs)")
    }

    protected def validateXMatchesY(x: DenseVector[Double], y: DenseVector[Double]) = {
      require(x.length == y.length, s"x.length(${x.length}) | y.length(${y.length})")
    }

    protected def doActivation(x: DenseVector[Double]): (Activation, Z) = {
      val z = thetas * DenseVector.vertcat(DenseVector(1d), x)
      val a = Sigmoid.sigmoid(z)
      (a, z)
    }

    protected[Layers] def activate(x: DenseVector[Double]): Result

    protected[Layers] def activateWithGradients(x: DenseVector[Double],
                                                y: DenseVector[Double]): (Result, Seq[Gradients], Delta)

  }

  trait ConnectableLayer extends Layer {

    var nextLayer: Option[Layer] = Option.empty

    override protected[Layers] def activate(x: DenseVector[Double]): Result = {
      require(nextLayer.isDefined)
      validateXInput(x)

      val (activation, _) = doActivation(x)
      nextLayer.get.activate(activation)
    }

    override protected[Layers] def activateWithGradients(x: DenseVector[Double],
                                                         y: DenseVector[Double]): (Result, Seq[Gradients], Delta) = {
      require(nextLayer.isDefined)
      validateXInput(x)

      val (activation, z) = doActivation(x)
      val (result, gradients, prevDelta) = nextLayer.get.activateWithGradients(activation, y)

      val curDelta = thetas.t * prevDelta *:* z
      val curGradient = prevDelta * activation.t
      (result, curGradient +: gradients, curDelta)
    }

    def connectTo(nextLayer: Layer): Unit = {
      require(nextLayer.inputs == this.units, s"nextLayer.inputs(${nextLayer.inputs}) | this.units($units)")
      this.nextLayer = Option(nextLayer)
    }

  }

  trait SourceLayer extends ConnectableLayer {

    override var thetas = DenseMatrix.zeros(0, 0)

    override def activate(x: DenseVector[Double]): Result = {
      require(nextLayer.isDefined)
      validateXInput(x)

      nextLayer.get.activate(x)
    }

    override def activateWithGradients(x: DenseVector[Double], y: DenseVector[Double]) = {
      require(nextLayer.isDefined)
      validateXInput(x)
      validateXMatchesY(x, y)

      nextLayer.get.activateWithGradients(x, y)
    }

  }

  trait SinkLayer extends Layer {

    override def activate(x: DenseVector[Double]): Result = {
      validateXInput(x)

      val (activation, _) = doActivation(x)

      activation
    }

    override def activateWithGradients(x: DenseVector[Double], y: DenseVector[Double]) = {
      validateXInput(x)

      val (activation, z) = doActivation(x)
      val delta = activation -:- z
      (activation, Seq.empty, delta)
    }


  }

}