package util

import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

/**
  *
  */
class GradientCheckerTest extends FunSpec with Matchers {
  Nd4j.setDataType(Type.DOUBLE)

  describe("The GradientChecker results") {

    it("should be very small differences") {
      val differences = GradientChecker.check()
      for (diff <- differences) {
        withClue("The absolute difference in the gradients should be very small") {
          diff._1 should be <= 0.1
        }
        withClue("The relative difference in the gradients should be very small") {
          diff._2 should be <= 0.01
        }
      }
    }

    it("should be very small differences with lambda") {
      val differences = GradientChecker.check(3)
      for (diff <- differences) {
        withClue("The absolute difference in the gradients should be very small") {
          diff._1 should be <= 0.1
        }
        withClue("The relative difference in the gradients should be very small") {
          diff._2 should be <= 0.01
        }
      }
    }

  }
}
