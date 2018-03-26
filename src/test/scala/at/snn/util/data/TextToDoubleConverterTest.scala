package at.snn.util.data

import at.snn.BaseMatcherTestClass

/**
  *
  */
class TextToDoubleConverterTest extends BaseMatcherTestClass {

  describe("TextToDoubleConverter") {
    val cut = new TextToDoubleConverter
    val value1 = "1"
    val value2 = "2"

    it("convertToDouble should return the same double for the same value") {
      val r1 = cut.convertToDouble(value1)
      val r2 = cut.convertToDouble(value1)

      r1 shouldBe r2
    }

    it("convertToDouble should not return the same double for different values") {
      val r1 = cut.convertToDouble(value1)
      val r2 = cut.convertToDouble(value2)

      r1 should not be r2
    }

  }
}
