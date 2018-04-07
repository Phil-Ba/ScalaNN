package at.snn.model.data

import at.snn.util.data.{FeatureNormalizer, LabelConverter}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Raw data implementation which dynamically expends, but might not be the most efficient one.
  *
  * @param columns of the x-data
  * @param labels  of the y-data
  */
class DynamicSizedRawData(val columns: Int, val labels: Int, val normalized: Boolean = false) extends RawData {

  private var x: INDArray = Nd4j.create(1, columns)
  private var y: INDArray = Nd4j.create(labels, 1)

  /**
    * Adds a new row of x-data
    *
    * @param data to add. Must be of size columns
    */
  def addData(data: Double*): Unit = {
    x = Nd4j.vstack(data.asNDArray(1, columns), x)
  }

  /**
    * Adds a new column of y-data
    *
    * @param label to add. Value must be < labels
    */
  def addLabel(label: Double): Unit = {
    val col = LabelConverter.labelToVector(label, labels)
    y = Nd4j.hstack(col, y)
  }

  /**
    * Gets to data collect so far as 2 INDArrays
    *
    * @return a tuple of (x-data, y-data)
    */
  def getData: (INDArray, INDArray) = {
    if (normalized == false) {
      (x, y)
    } else {
      (FeatureNormalizer.normalize(x)._1, y)
    }
  }

}