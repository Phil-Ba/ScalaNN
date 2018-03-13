package at.snn.util.data

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits.doubleArray2INDArray

import scala.collection.immutable.HashMap
import scala.io.Codec

/**
  *
  */
object MatlabImporter {

  def apply(file: String): HashMap[String, INDArray] = {
    val lines = scala.io.Source.fromFile(file)(Codec.ISO8859)
      .getLines()

    var data = HashMap[String, INDArray]()

    while (lines.hasNext) {
      readHeaders(lines).foreach(header => {
        val (name, _, rows, columns) = header
        data += (name -> readData(rows, columns, lines))
      })
    }

    data
  }

  private def readHeaders(lines: Iterator[String]): Option[(String, String, Int, Int)] = {
    val headers = lines.dropWhile(_.startsWith("# name:") == false).take(4)
    if (headers.isEmpty) {
      None
    } else {
      val name = headers.next.drop(8)
      val dataType = headers.next.drop(8)
      val rows = headers.next.drop(8).toInt
      val columns = headers.next.drop(11).toInt
      Option(name, dataType, rows, columns)
    }
  }

  private def readData(rows: Int, columns: Int, lines: Iterator[String]): INDArray = {
    val data = lines.take(rows)
      .flatMap(row => row.tail.split(' ').map(_.toDouble))
      .toArray
    data.asNDArray(rows, columns)
  }

}
