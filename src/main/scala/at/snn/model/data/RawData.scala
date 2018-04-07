package at.snn.model.data

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * General abstraction for raw data which is provided by some input source
  *
  */
trait RawData {
  val columns: Int
  val labels: Int
  val normalized: Boolean


  /**
    * Adds a new row of x-data
    *
    * @param data to add. Must be of size columns
    */
  def addData(data: Double*): Unit

  /**
    * Adds a new column of y-data
    *
    * @param label to add. Value must be < labels
    */
  def addLabel(label: Double): Unit
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
  def getData: (INDArray, INDArray)

}