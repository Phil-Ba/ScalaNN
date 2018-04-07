package at.snn.model.data

import at.snn.util.data.{FeatureNormalizer, LabelConverter}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * RawData implementation which is fixed sized, which might be better from a memory/performance standpoint.
  *
  * @param columns of the x-data
  * @param labels  of the y-data
  */
class FixedSizedRawData(dataSize: Int, val columns: Int, val labels: Int, val normalized: Boolean = false) extends RawData {

  private val x: INDArray = Nd4j.createUninitialized(dataSize, columns)
  private val y: INDArray = Nd4j.createUninitialized(labels, dataSize)
  private var rowCount = 1

  /**
    * Adds a new row of x-data
    *
    * @param data to add. Must be of size columns
    */
  override def addData(data: Double*): Unit = {
    for (i <- 0 until columns) {
      x(rowCount, i) = data(i)
    }
  }

  /**
    * Adds a new column of y-data
    *
    * @param label to add. Value must be < labels
    */
  override def addLabel(label: Double): Unit = {
    val col = LabelConverter.labelToVector(label, labels)
    y(->, rowCount) = col
  }

  /**
    * Gets to data collect so far as 2 INDArrays
    *
    * @return a tuple of (x-data, y-data)
    */
  override def getData: (INDArray, INDArray) = {
    if (normalized == false) {
      (x, y)
    } else {
      (FeatureNormalizer.normalize(x)._1, y)
    }
  }

}