package at.snn.model.data

import at.snn.util.data.LabelConverter
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * General abstraction for raw data which is provided by some input source
  *
  * @param columns of the x-data
  * @param labels  of the y-data
  */
class RawData(columns: Int, labels: Int) {

  private var x: Seq[INDArray] = Nil
  private var y: Seq[INDArray] = Nil

  /**
    * Adds a new row of x-data
    *
    * @param data to add. Must be of size columns
    */
  def addData(data: Double*): Unit = {
    x = data.asNDArray(1, columns) +: x
  }

  /**
    * Adds a new column of y-data
    *
    * @param label to add. Value must be < labels
    */
  def addLabel(label: Double): Unit = {
    val col = LabelConverter.labelToVector(label, labels)
    y = col +: y
  }

  /**
    * Adds x- and y-data simultaneously
    *
    * @param data  to add
    * @param label to add
    */
  def addDataAndLabel(data: Double*)(label: Double): Unit = {
    addData(data: _*)
    addLabel(label)
  }

  /**
    * Gets to data collect so far as 2 INDArrays
    *
    * @return a tuple of (x-data, y-data)
    */
  def getData: (INDArray, INDArray) = {
    (Nd4j.vstack(x: _*), Nd4j.hstack(y: _*))
  }

}