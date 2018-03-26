package at.snn.util.data

import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.{FunSpec, Matchers}

import scala.util.Random


/**
  *
  */
class FeatureNormalizerTest extends FunSpec with Matchers {

  Nd4j.setDataType(Type.DOUBLE)
  private val feature1 = Array(2.5, 4.5, 3.8, 1.2)
  private val feature2 = Array(25D, 15D, 12D, 8D)
  private val feature3 = Array(3D, 7D, 4D, 6D)
  private val testData = Nd4j.vstack(feature1.toNDArray, feature2.toNDArray, feature3.toNDArray)

  describe("The FeatureNormalizer normalize method") {
    it("should return a normalized dataset") {
      val (normDataset, mu, sigma) = FeatureNormalizer.normalize(testData)

      for (i <- 0 until 3) {
        Nd4j.mean(normDataset(i, ->)).getDouble(0) shouldBe 0.0 +- 0.001 withClue s"the mean of the $i-th feature should be close to 0"
        Nd4j.std(normDataset(i, ->)).getDouble(0) shouldBe 1.0 +- 0.02 withClue s"the std of the $i-th feature should be close to 1"
      }
    }
    it("should return correct mu and sigma values") {
      val newData = (1 to 9).map(_ => Random.nextDouble() * 100).asNDArray(3, 3)

      val (normDataset, mu, sigma) = FeatureNormalizer.normalize(testData)
      val validationNormDataset = FeatureNormalizer.normalize(testData, mu, sigma)
      validationNormDataset shouldBe normDataset withClue "For correct mu/sigma values the normalization should return the same values as" +
        " originally"
    }
  }

}
