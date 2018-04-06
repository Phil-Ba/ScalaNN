package at.snn.util.data

import java.io.File

import at.snn.model.data.RawData
import com.univocity.parsers.common.processor.RowListProcessor
import com.univocity.parsers.csv.{CsvParser, CsvParserSettings}

import scala.collection.convert.WrapAsScala.asScalaBuffer

/**
  *
  */
object WineImporter {

  def apply(file: String): RawData = {
    val rowProcessor = new RowListProcessor()

    val settings = new CsvParserSettings()
    settings.setLineSeparatorDetectionEnabled(true)
    settings.setDelimiterDetectionEnabled(true)
    settings.setQuoteDetectionEnabled(true)
    settings.setHeaderExtractionEnabled(true)
    settings.setProcessor(rowProcessor)

    val parser = new CsvParser(settings)
    parser.parse(new File(file))

    val rawData = new RawData(11, 10)
    rowProcessor
      .getRows
      .foreach(row => {
        val rowAsDoubles = row.map(_.toDouble)
        rawData.addDataAndLabel(rowAsDoubles.slice(0, 12): _*)(rowAsDoubles.last)
      })

    rawData
  }
}