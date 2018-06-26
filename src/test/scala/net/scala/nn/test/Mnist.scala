package net.scala.nn.test

import java.io.File
import java.net.URL
import java.nio.file.Files

/**
  * Created by wangmich on 06/26/2018.
  */
object Mnist extends App {
  val mnistFiles =("train" -> "t10k-images-idx3-ubyte",
                    "train-label" -> "t10k-labels-idx1-ubyte",
                    "test" -> "train-images-idx3-ubyte",
                    "test-label" -> "train-labels-idx1-ubyte")

  println(new File("t10k-images-idx3-ubyte").exists())

}
