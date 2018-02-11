name := "ScalaNN"

version := "0.1"

scalaVersion := "2.11.12"

resolvers += Resolver.mavenLocal

val breezeVersion = "0.13.+"
val nd4jVersion = "0.9.+"
val tinyLogVersion = "1.3"

libraryDependencies ++= Seq(
  //  "org.scalanlp" %% "breeze" % breezeVersion,
  //  "org.scalanlp" %% "breeze-natives" % breezeVersion,
  //  "org.scalanlp" %% "breeze-viz" % breezeVersion,

  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
  "org.nd4j" % "nd4s_2.11" % nd4jVersion,

  "org.jfree" % "jfreechart" % "1.5.+",

  "com.typesafe.scala-logging" %% "scala-logging" % "3.7.+",
  "ch.qos.logback" % "logback-classic" % "1.2.+",
  //"org.tinylog" % "tinylog" % tinyLogVersion,
  //"org.tinylog" % "slf4j-binding" % tinyLogVersion,

  "org.scalatest" %% "scalatest" % "3.0.+" % "test"
)