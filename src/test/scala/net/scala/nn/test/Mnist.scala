package net.scala.nn.test

import scala.collection.mutable.ListBuffer
import java.io.DataInputStream

import java.nio.file.Files

import breeze.linalg.{DenseVector, DenseMatrix}

import scala.io.Source

/**
  * Created by wangmich on 06/26/2018.
  */
object Mnist extends App {
  val mnistFiles =("train" -> "t10k-images-idx3-ubyte",
                    "train-label" -> "t10k-labels-idx1-ubyte",
                    "test" -> "train-images-idx3-ubyte",
                    "test-label" -> "train-labels-idx1-ubyte")

  val stream = getClass.getResourceAsStream("/t10k-labels-idx1-ubyte")
  val dataStream  = new DataInputStream(stream)
  val labels = ListBuffer.empty[DenseVector[Double]]

  println(dataStream.readInt)
  val count = dataStream.readInt()
  println(count)

  for (c <- 0 until count) {
    val label = dataStream.readByte()
    labels += DenseVector.tabulate[Double](10)({ i => if (i == label) 1.0 else 0.0 })
  }
  println(labels.toList)
  dataStream.close()

  val imageStream = getClass.getResourceAsStream("/t10k-images-idx3-ubyte")
  val imageDataStream  = new DataInputStream(imageStream)
  val images = ListBuffer.empty[DenseVector[Double]]
  println(imageDataStream.readInt())

  val imageCount = imageDataStream.readInt()
  val height = imageDataStream.readInt()
  val width = imageDataStream.readInt()

  println(s"imageCount=${imageCount}, height=${height}, width=${width}")

  for (c <- 0 until imageCount) {
    val matrix = DenseMatrix.zeros[Int](height, width)
    for (r <- 0 until height; c <- 0 until width) {
      matrix(r, c) = imageDataStream.readUnsignedByte()
    }
 //   println(matrix)
    images += DenseVector.tabulate(height * width)({ i => matrix(i / width, i % height) / 255.0 })
  }

  //println(images.toList)
  imageDataStream.close
}
