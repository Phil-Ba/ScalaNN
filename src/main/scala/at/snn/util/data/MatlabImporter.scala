package at.snn.util.data

import at.snn.model.data.RawData

import scala.io.Codec

/**
  *
  */
object MatlabImporter {

  def apply(file: String): RawData = {
    val lines = scala.io.Source.fromFile(file)(Codec.ISO8859)
      .getLines()

    val data = new RawData(255, 10)

    while (lines.hasNext) {
      readHeaders(lines).foreach(header => {
        val (name, _, rows, columns) = header
        val read = readData(rows, columns, lines)
        if (name == "X") {
          read
            .sliding(columns, columns)
            .foreach(line => data.addData(line: _*))
        } else {
          read.foreach(data.addLabel)
        }

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

  private def readData(rows: Int, columns: Int, lines: Iterator[String]): Iterator[Double] = {
    lines
      .take(rows)
      .flatMap(row => row.tail.split(' ').map(_.toDouble))
  }

}
