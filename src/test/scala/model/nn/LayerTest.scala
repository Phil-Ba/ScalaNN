package model.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.prop.TableDrivenPropertyChecks
import org.scalatest.{FunSpec, Matchers}

/**
  *
  */
class LayerTest extends FunSpec with Matchers with TableDrivenPropertyChecks {

  describe("A simple AND neuronal network") {
    val andTable = Table(("x1", "x2", "expected"),
      (1, 1, 1),
      (1, 0, 0),
      (0, 1, 0),
      (0, 0, 0)
    )
    val in = new InputLayer(2)
    val out = new OutputLayer(DenseMatrix((-40d, 30d, 30d)))
    in.connectTo(out)

    forAll(andTable) { (x1: Int, x2: Int, expected: Int) =>
      it(s"should produce $expected for input x1($x1) AND x2($x2)") {
        val result = in.activate(DenseVector(x1, x2))
        result.length shouldBe 1
        result(0) shouldBe expected.toDouble +- 0.001
      }
    }
  }

}
