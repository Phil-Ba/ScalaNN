package at.snn.util.plot

import com.typesafe.scalalogging.StrictLogging
import org.jfree.chart.{ChartFactory, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}

/**
  *
  */
object PlotCost extends StrictLogging {

  def plot(data: Seq[(String, Seq[Double])]): JFreeChart = {
    val seriesCollection = new XYSeriesCollection
    data.map { case (name, costs) =>
      val series = new XYSeries(name)
      costs.zipWithIndex.foreach {
        case (cost, idx) => series.add(idx, cost)
      }
      series
    }.foreach(series => seriesCollection.addSeries(series))

    val chart = ChartFactory.createXYLineChart("Costs", "Iteration", "Cost", seriesCollection)
    logger.info("Finished cost plot!")
    chart
  }

}
