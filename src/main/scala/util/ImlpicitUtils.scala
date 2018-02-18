package util

import org.nd4j.linalg.api.ndarray.INDArray

/**
  *
  */
object ImlpicitUtils {

  implicit class DebugINDArray[T <: INDArray](array: T) {

    def printShape: String = s"${array.rows()}:${array.columns()}"

  }

}
