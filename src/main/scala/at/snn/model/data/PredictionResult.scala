package at.snn.model.data

/**
  *
  */
case class PredictionResult(totalPredictions: Int, correctPredictions: Int, wrongPredictions: Int) {

  val correctPercent: Double = (correctPredictions / totalPredictions.toDouble) * 100
  val wrongPercent: Double = (wrongPredictions / totalPredictions.toDouble) * 100

}

object PredictionResult {
  def apply(predictions: Seq[(Boolean, Int, Int)]): PredictionResult = {
    val totalPredictions = predictions.size
    val correctPredictions = predictions.count(_._1)
    val wrongPredictions = predictions.count(_._1 == false)
    new PredictionResult(totalPredictions, correctPredictions, wrongPredictions)
  }
}
