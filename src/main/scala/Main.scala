/**
  *
  */
object Main {

  def main(args: Array[String]): Unit = {
    val map = MatlabImporter("src/main/resources/test.mat")
    println(map)
  }

}
