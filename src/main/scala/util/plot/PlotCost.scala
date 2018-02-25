package util.plot

import org.jfree.chart.{ChartFactory, JFreeChart}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}

/**
  *
  */
object PlotCost {

  def plot(data: Seq[(String, Seq[Double])]): JFreeChart = {
    val seriesCollection = new XYSeriesCollection
    data.map { case (name, costs) =>
      val series = new XYSeries(name)
      costs.zipWithIndex.foreach {
        case (cost, idx) => series.add(idx, cost)
      }
      series
    }.foreach(series => seriesCollection.addSeries(series))
    ChartFactory.createXYLineChart("Costs", "Iteration", "Cost", seriesCollection)
  }

}
