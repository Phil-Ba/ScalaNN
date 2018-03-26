package at.snn.util.data

import at.snn.model.Constants
import org.nd4j.linalg.api.ndarray.INDArray

object FeatureNormalizer {

  /**
    * Normalizes all values from a dataset. Features are assumed to be arranged column-wise, ie. each column
    * represents a separate feature.
    *
    * @param dataset to normalize
    * @return a tuple containing, (the normalized dataset, mu of each feature as column vector, sigma of each feature as column vector)
    */
  def normalize(dataset: INDArray): (INDArray, INDArray, INDArray) = {
    val mu = dataset.mean(Constants.DIMENSION_DIRECTION_COLUMN)
    val sigma = dataset.std(Constants.DIMENSION_DIRECTION_COLUMN)

    val result = normalize(dataset, mu, sigma)

    (result, mu, sigma)
  }

  /**
    * Normalizes the input according to the given mu and sigma values.
    *
    * @param inputToNormalize these values will be normalized
    * @param mu               column vector containing a mu value for each column of #inputToNormalize
    * @param sigma            column vector containing a sigma value for each column of #inputToNormalize
    * @return the normalized values
    */
  def normalize(inputToNormalize: INDArray, mu: INDArray, sigma: INDArray): INDArray = {
    inputToNormalize.subColumnVector(mu).diviColumnVector(sigma)
  }

}
