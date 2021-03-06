package at.snn.model.nn

import at.snn.BaseMatcherTestClass
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.prop.TableDrivenPropertyChecks

/**
  *
  */
class LayerTest extends BaseMatcherTestClass with TableDrivenPropertyChecks {

  describe("A simple AND neuronal network") {
    val andTable = Table(("x1", "x2", "expected"),
      (1, 1, 1),
      (1, 0, 0),
      (0, 1, 0),
      (0, 0, 0)
    )
    val in = new InputLayer(2)
    val out = new OutputLayer(Array(-40d, 30d, 30d).toNDArray)
    in.connectTo(out)

    forAll(andTable) { (x1: Int, x2: Int, expected: Int) =>
      it(s"should produce $expected for input x1($x1) AND x2($x2)") {
        val result = in.activate(Array(x1.toDouble, x2.toDouble).toNDArray)
        val (r, g, _) = in.activateWithGradients(Array(x1, x2).toNDArray, Nd4j.ones(1))
        println(r)
        println(g)
        result.length shouldBe 1
        result(0) shouldBe expected.toDouble +- 0.001
      }
    }
  }

}
