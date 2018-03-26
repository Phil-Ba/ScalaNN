package at.snn.util.data

import scala.collection.mutable

/**
  * Converts textual values into double values, by assigning the first occurrence of a string a number.
  * All subsequent occurrences will receive the same number value. This will only be useful if your values are
  * some enumerated strings.
  */
class TextToDoubleConverter {

  private val values: mutable.Map[String, Double] = new mutable.HashMap[String, Double]

  def convertToDouble(inp: String): Double = {
    values.getOrElseUpdate(inp, values.maxBy(_._2)._2 + 1)
  }

}
