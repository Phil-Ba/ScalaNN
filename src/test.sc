import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import util.{CostFunction, RandomInitializier}

Nd4j.setDataType(DataBuffer.Type.DOUBLE)
var m = Nd4j.zeros(3, 2)
m = Nd4j.hstack(Nd4j.ones(m.rows(), 1), m)
m = (1 to 12).asNDArray(4, 3)

m(--->)

m(1 -> m.rows(), ->)

"---"

m = Nd4j.zeros(25, 1)
Nd4j.vstack(Nd4j.ones(1, 1), m)

val n = Nd4j.zeros(3, 1)
n.shape()
n.T.shape()
val i = 1
m.rsub(1)

1000 / 100 * 10
val mm = (1 to 9).asNDArray(1, 9)
mm.reshape('f', 3, 3)




RandomInitializier.initialize(5, 5)

Array(1, 0, 1, 1, 0, 1, 0, 0).asNDArray(3, 2)

val y1 = Array(
  Array(0, 1, 1),
  Array(1, 0, 1),
  Array(1, 1, 0)
).toNDArray
val y2 = Array(
  Array(0, 1, 1),
  Array(1, 0, 1),
  Array(1, 1, 0)
).toNDArray

val x = Array(
  0.00011266153022739,
  0.00174127855749204,
  0.00252696958994872,
  0.000018403232103925,
  0.00936263859928631,
  0.00399270267069424,
  0.00551517523779625,
  0.000401468104906522,
  0.00648072305307484,
  0.995734011986168
).asNDArray(10, 1)

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
).asNDArray(10, 1)

"cost1"
CostFunction.cost(x, y)



"cost2"
val l1 = Transforms.log(x)
val l2 = Transforms.log(x.rsub(1))
(y.neg() dot l1 - (y.rsub(1)) dot l2)
