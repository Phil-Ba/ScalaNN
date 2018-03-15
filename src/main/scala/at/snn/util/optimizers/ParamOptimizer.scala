package at.snn.util.optimizers

import at.snn.model.data.SampleSet
import at.snn.model.nn.InputLayer
import at.snn.util.{CostFunction, NNRunner}

/**
  *
  */
object ParamOptimizer {

  def optimizeByPredictions(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double) => Any,
                            paramValues: Seq[Double]): (Double, Double) = {
    paramValues.map { p =>
      val nn = network.copyNetwork
      function(nn, p)
      val correctPercent = NNRunner.runPredictions(nn, dataSet.cvSet, dataSet.cvResultSet).correctPercent
      (p, correctPercent)
    }.maxBy(_._2)
  }

  def optimizeByPredictions(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double, Double) => Any,
                            param1Values: Seq[Double], param2Values: Seq[Double]): (Double, Double, Double) = {
    val results = for {
      p1 <- param1Values
      p2 <- param2Values
    } yield {
      val nn = network.copyNetwork
      function(nn, p1, p2)
      val correctPercent = NNRunner.runPredictions(nn, dataSet.cvSet, dataSet.cvResultSet).correctPercent
      (p1, p2, correctPercent)
    }
    results.maxBy(_._3)
  }

  def optimizeByCost(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double) => Any,
                     paramValues: Seq[Double]): (Double, Double) = {
    paramValues.map { p =>
      val nn = network.copyNetwork
      function(nn, p)
      val calcY = NNRunner.runWithData(dataSet.cvSet, nn)
      val cost = CostFunction.cost(calcY, dataSet.cvResultSet)
      (p, cost)
    }.minBy(_._2)
  }

  def optimizeByCost(dataSet: SampleSet, network: InputLayer, function: (InputLayer, Double, Double) => Any,
                     param1Values: Seq[Double], param2Values: Seq[Double]): (Double, Double, Double) = {
    val results = for {
      p1 <- param1Values
      p2 <- param2Values
    } yield {
      val nn = network.copyNetwork
      function(nn, p1, p2)
      val calcY = NNRunner.runWithData(dataSet.cvSet, nn)
      val cost = CostFunction.cost(calcY, dataSet.cvResultSet)
      (p1, p2, cost)
    }
    results.minBy(_._3)
  }
}
