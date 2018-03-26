package at.snn

import org.nd4j.linalg.api.buffer.DataBuffer.Type
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FunSpec, Matchers}

/**
  *
  */
trait BaseMatcherTestClass extends FunSpec with Matchers {

  Nd4j.setDataType(Type.DOUBLE)


}
