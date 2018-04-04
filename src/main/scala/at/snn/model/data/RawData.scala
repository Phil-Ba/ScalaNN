package at.snn.model.data

import at.snn.util.data.LabelConverter
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
class RawData(columns: Int, labels: Int) {

  private var x: Seq[INDArray] = Nil
  private var y: Seq[INDArray] = Nil

  def addData(data: Double*): Unit = {
    x = data.asNDArray(1, columns) +: x
  }

  def addLabel(label: Double): Unit = {
    val col = LabelConverter.labelToVector(label, labels)
    y = col +: y
  }

  def addDataAndLabel(data: Double*)(label: Double): Unit = {
    addData(data: _*)
    addLabel(label)
  }

  def getData(): (INDArray, INDArray) = {
    (Nd4j.vstack(x: _*), Nd4j.hstack(y: _*))
  }

}