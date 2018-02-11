package util.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  */
object LabelConverter {

  def labelToVector(label: Int, labels: Int): INDArray = {
    val labelIdx = label % labels
    val vector = Nd4j.zeros(labels, 1)
    vector(labelIdx, 0) = 1
    vector
  }

  def labelToVector(labelVector: INDArray, label: Int, labels: Int) = {
    val labelIdx = label % labels
    labelVector(labelIdx, 0) = 1
  }

  def vectorToLabel(vector: INDArray): Int = {
    val sorted = Nd4j.sortWithIndices(vector, 0, false)
    val label = sorted(0)(0)
    label.toInt
  }

}