package at.snn.util.optimizers

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.{CostFunction, NNRunner}

/**
  *
  */
object ParamOptimizer {

  def optimizeByPredictions(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double) => Any,
                            from: Double = 0.5, to: Double = 1.0, stepSize: Double = 0.3): (Double, Double) = {
    (from to to by stepSize).map { p =>
      val nn = network.copyNetwork
      function(nn, p)
      val correctPercent = NNRunner.runPredictions(nn, dataSet.cvSet, dataSet.cvResultSet).correctPercent
      (p, correctPercent)
    }.maxBy(_._2)
  }

  def optimizeByCost(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double) => Any,
                     from: Double = 0.5, to: Double = 1.0, stepSize: Double = 0.3): (Double, Double) = {
    (from to to by stepSize).map { p =>
      val nn = network.copyNetwork
      function(nn, p)
      val calcY = NNRunner.runWithData(dataSet.cvSet, nn)
      val cost = CostFunction.cost(calcY, dataSet.cvResultSet)
      (p, cost)
    }.maxBy(_._2)
  }

}
