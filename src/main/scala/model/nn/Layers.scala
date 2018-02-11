package model.nn

import com.typesafe.scalalogging.StrictLogging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

object Layers {

  trait Layer extends StrictLogging {
    type Result = INDArray
    type Delta = INDArray
    type Activation = INDArray
    type Z = INDArray
    type Gradients = INDArray
    type Thetas = INDArray

    protected[Layers] val thetas: Thetas
    val units: Int
    val inputs: Int

    protected def validateXInput(x: INDArray) = {
      require(x.columns() == this.inputs, s"x.cols(${x.columns()}) | this.inputs($inputs)")
    }

    protected def doActivation(x: INDArray): (Activation, Z) = {
      val xPlusBias = Nd4j.hstack(Nd4j.ones(x.rows, 1), x)
      //units X inputs   inputs X samples
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
      logger.debug("{} got input[{}] and produced output[{}]", this, x.shape().mkString("x"), activation.shape().mkString("x"))

      nextLayer.get.activate(activation.T)
    }

    override protected[Layers] def activateWithGradients(x: INDArray,
                                                         y: INDArray): (Result, Seq[Gradients], Delta) = {
      require(nextLayer.isDefined)
      validateXInput(x)
      //25x401
      val (activation, z) = doActivation(x)
      val (result, gradients, prevDelta) = nextLayer.get.activateWithGradients(activation.T, y)
      val nextThetas = nextLayer.get.thetas
      logger.debug("theta[{}]", thetas.shapeInfoToString())
      logger.debug("prevDelta[{}]", prevDelta.shapeInfoToString())
      logger.debug("z[{}]", z.shapeInfoToString())
      logger.debug("nextThetas[{}]", nextLayer.get.thetas.shapeInfoToString())
      //26x10 10x1 25x1
      val curDelta = (nextThetas(->, 1 -> nextThetas.columns()).T dot prevDelta) * Transforms.sigmoidDerivative(z, true)
      val curGradient = curDelta dot Nd4j.hstack(Nd4j.ones(1, 1), x)
      (result, curGradient +: gradients, curDelta)
    }

    def connectTo(nextLayer: Layer): Unit = {
      require(nextLayer.inputs == this.units, s"nextLayer.inputs(${nextLayer.inputs}) | this.units($units)")
      this.nextLayer = Option(nextLayer)
    }

  }

  trait SourceLayer extends ConnectableLayer {

    private lazy val layers = collectLayers(nextLayer)

    def updateWithGradients(gradients: Seq[Gradients]): Unit = {
      gradients.zip(getNNThetas).foreach { gt =>
        gt._2 -= gt._1
      }
    }

    private def collectLayers(layer: Option[Layer]): Seq[Layer] = {
      layer.fold(Seq.empty[Layer]) {
        case cl: ConnectableLayer => cl +: collectLayers(cl.nextLayer)
        case l => Seq(l)
      }
    }

    def getNNThetas: Seq[INDArray] = {
      layers.map(_.thetas)
    }

    override protected def validateXInput(x: Thetas): Unit = {
      require(x.columns() == this.inputs, s"x.cols(${x.columns}) | this.inputs($inputs)")
    }

    override protected[Layers] val thetas: Thetas = null

    override def activate(x: INDArray): Result = {
      require(nextLayer.isDefined)
      validateXInput(x)

      nextLayer.get.activate(x)
    }

    override def activateWithGradients(x: INDArray, y: INDArray): (Result, Seq[Gradients], Delta) = {
      require(nextLayer.isDefined)
      validateXInput(x)

      nextLayer.get.activateWithGradients(x, y)
    }

    def activateWithGradients(x: INDArray, y: INDArray, lambda: Double): (Result, Seq[Gradients], Delta) = {
      val (result, gradients, delta) = activateWithGradients(x, y)
      if (lambda == 0) {
        (result, gradients, delta)
      }
      else {
        val penalizedGradients = gradients.zip(getNNThetas)
          .map { gt =>
            val gradientsExclBias = gt._1(->, 1 until gt._1.columns())
            val thetasExclBias = gt._2(->, 1 until gt._2.columns())
            val penalizedGradients = gradientsExclBias + (thetasExclBias * lambda)
            Nd4j.hstack(gt._1(->, 0), penalizedGradients)
          }
        (result, penalizedGradients, delta)
      }
    }

  }

  trait SinkLayer extends Layer {

    override protected[Layers] def activate(x: INDArray): Result = {
      validateXInput(x)

      val (activation, _) = doActivation(x)
      logger.debug("{} got input[{}] and produced output[{}]", this, x.shape().mkString("x"), activation.shape().mkString("x"))

      activation
    }

    override protected[Layers] def activateWithGradients(x: INDArray,
                                                         y: INDArray): (Result, Seq[Gradients], Delta) = {
      validateXInput(x)

      val (activation, _) = doActivation(x)

      val delta = activation - y
      logger.debug("X[{}]", x)
      logger.debug("Y[{}]", y)
      logger.debug("Activation[{}]", activation)
      logger.debug("Delta[{}]", delta)
      //      10x1 1x25
      val curGradient = delta dot Nd4j.hstack(Nd4j.ones(1, 1), x)

      (activation, Seq(curGradient), delta)
    }


  }

}