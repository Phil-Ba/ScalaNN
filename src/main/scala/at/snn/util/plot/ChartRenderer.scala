package at.snn.util.plot

import javax.swing.{JFrame, SwingUtilities, WindowConstants}

import org.jfree.chart.{ChartPanel, JFreeChart}

/**
  *
  */
object ChartRenderer {

  def render(chart: JFreeChart): Unit = {
    val runnable = new Runnable() {
      override def run(): Unit = {
        val panel = new ChartPanel(chart)
        val frame = new JFrame()
        val factor = 100
        frame.setSize(16 * factor, 9 * factor)
        frame.setLocationRelativeTo(null)
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        frame.setContentPane(panel)
        frame.setVisible(true)
      }
    }
    SwingUtilities.invokeLater(runnable)
  }

}
