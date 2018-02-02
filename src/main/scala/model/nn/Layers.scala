package model.nn

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

object Layers {

  trait Layer {
    type Result = INDArray
    type Delta = INDArray
    type Activation = INDArray
    type Z = INDArray
    type Gradients = INDArray
    type Thetas = INDArray

    protected var thetas: Thetas
    val units: Int
    val inputs: Int

    protected def validateXInput(x: INDArray) = {
      require(x.columns() == this.inputs, s"x.cols(${x.columns()}) | this.inputs($inputs)")
    }

    protected def validateXMatchesY(x: INDArray, y: INDArray) = {
      require(x.columns() == y.columns(), s"x.cols(${x.columns()}) | y.cols(${y.columns()})")
    }

    protected def doActivation(x: INDArray): (Activation, Z) = {
      val xPlusBias = Nd4j.hstack(Nd4j.ones(x.rows, 1), x)
      val z = thetas dot xPlusBias.T
      val a = Transforms.sigmoid(z)
      (a, z)
    }

    protected[Layers] def activate(x: INDArray): Result

    protected[Layers] def activateWithGradients(x: INDArray,
                                                y: INDArray): (Result, Seq[Gradients], Delta)

  }

  trait ConnectableLayer extends Layer {

    var nextLayer: Option[Layer] = Option.empty

    override protected[Layers] def activate(x: INDArray): Result = {
      require(nextLayer.isDefined)
      validateXInput(x)

      val (activation, _) = doActivation(x)
      nextLayer.get.activate(activation)
    }

    override protected[Layers] def activateWithGradients(x: INDArray,
                                                         y: INDArray): (Result, Seq[Gradients], Delta) = {
      require(nextLayer.isDefined)
      validateXInput(x)

      val (activation, z) = doActivation(x)
      val (result, gradients, prevDelta) = nextLayer.get.activateWithGradients(activation, y)

      val curDelta = thetas.T dot prevDelta * z
      val curGradient = prevDelta dot activation.T
      (result, curGradient +: gradients, curDelta)
    }

    def connectTo(nextLayer: Layer): Unit = {
      require(nextLayer.inputs == this.units, s"nextLayer.inputs(${nextLayer.inputs}) | this.units($units)")
      this.nextLayer = Option(nextLayer)
    }

  }

  trait SourceLayer extends ConnectableLayer {

    override protected var thetas: Thetas = _

    override def activate(x: INDArray): Result = {
      require(nextLayer.isDefined)
      validateXInput(x)

      nextLayer.get.activate(x)
    }

    override def activateWithGradients(x: INDArray,
                                       y: INDArray): (Result, Seq[Gradients], Delta) = {
      require(nextLayer.isDefined)
      validateXInput(x)
      validateXMatchesY(x, y)

      nextLayer.get.activateWithGradients(x, y)
    }

  }

  trait SinkLayer extends Layer {

    override protected[Layers] def activate(x: INDArray): Result = {
      validateXInput(x)

      val (activation, _) = doActivation(x)

      activation
    }

    override protected[Layers] def activateWithGradients(x: INDArray,
                                                         y: INDArray): (Result, Seq[Gradients], Delta) = {
      validateXInput(x)

      val (activation, z) = doActivation(x)
      val delta = activation - z
      (activation, Seq.empty, delta)
    }


  }

}