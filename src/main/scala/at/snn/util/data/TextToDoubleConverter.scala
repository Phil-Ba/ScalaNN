package at.snn.util.data

import scala.collection.mutable

/**
  * Converts textual values into double values, by assigning the first occurrence of a string a number.
  * All subsequent occurrences will receive the same number value. This will only be useful if your values are
  * some enumerated strings.
  */
class TextToDoubleConverter {

  private val values: mutable.Map[String, Double] = new mutable.HashMap[String, Double]

  /**
    *
    * @param inp string to convert
    * @return the double value the string is mapped to
    */
  def convertToDouble(inp: String): Double = {
    values.getOrElseUpdate(inp, {
      if (values.isEmpty) {
        0.0
      } else {
        values.values.max + 1
      }
    })
  }

}
