package at.snn.model.data

import org.nd4j.linalg.api.ndarray.INDArray

/**
  *
  */
case class SampleSet(trainingSet: INDArray, trainingResultSet: INDArray,
                     cvSet: INDArray, cvResultSet: INDArray,
                     testSet: INDArray, testResultSet: INDArray) {

}