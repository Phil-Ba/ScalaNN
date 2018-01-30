name := "ScalaNN"

version := "0.1"

scalaVersion := "2.12.4"

resolvers += Resolver.mavenLocal

val breezeVersion = "0.13.+"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion,
  "org.scalanlp" %% "breeze-viz" % breezeVersion,

  "org.scalatest" %% "scalatest" % "3.0.+" % "test"
)