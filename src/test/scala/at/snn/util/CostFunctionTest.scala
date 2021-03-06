package at.snn.util

import at.snn.BaseMatcherTestClass
import org.nd4s.Implicits._

/**
  *
  */
class CostFunctionTest extends BaseMatcherTestClass {

  describe("The cost function") {
    it("should calculate the correct cost given these sample input values") {
      val x = Array(
        0.000112661530227,
        0.001741278557492,
        0.002526969589948,
        0.000018403232103,
        0.009362638599286,
        0.003992702670694,
        0.005515175237796,
        0.000401468104906,
        0.006480723053074,
        0.995734011986168
        ,
        0.976646079785914,
        0.0093585665802756,
        0.00831239004778868,
        0.000481286620014635,
        0.000238152582560706,
        0.00740703780978473,
        0.000425146661782421,
        0.0880166928274687,
        0.000687255476369219,
        0.000434284478978731

      ).asNDArray(10, 2)

      val y = Array(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1
        ,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ).asNDArray(10, 2)

      val cost = CostFunction.cost(x, y)
      cost shouldBe 0.08886799398566353 +- 0.00001
    }
  }

}
