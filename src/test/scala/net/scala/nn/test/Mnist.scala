package net.scala.nn.test

import scala.collection.mutable.ListBuffer
import java.io.DataInputStream
import breeze.linalg.{DenseVector, DenseMatrix}
import net.scala.nn.network

/**
  * Created by wangmich on 06/26/2018.
  */
object Mnist extends App {
  val mnistFiles =("train" -> "t10k-images-idx3-ubyte",
                    "train-label" -> "t10k-labels-idx1-ubyte",
                    "test" -> "train-images-idx3-ubyte",
                    "test-label" -> "train-labels-idx1-ubyte")

  val stream = getClass.getResourceAsStream("/train-labels-idx1-ubyte")
  val dataStream  = new DataInputStream(stream)
  val labels = ListBuffer.empty[DenseVector[Double]]

  println(dataStream.readInt)
  val count = dataStream.readInt() //skip  magic number
  println(count)

  for (c <- 0 until count) {
    val label = dataStream.readByte()
    labels += DenseVector.tabulate[Double](10)({ i => if (i == label) 1.0 else 0.0 })
  }
  println(labels.toList)
  dataStream.close()

  val imageStream = getClass.getResourceAsStream("/train-images-idx3-ubyte")
  val imageDataStream  = new DataInputStream(imageStream)
  val images = ListBuffer.empty[DenseVector[Double]]
  println(imageDataStream.readInt()) //skip magic number

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

  val teststream = getClass.getResourceAsStream("/t10k-labels-idx1-ubyte")
  val testdataStream  = new DataInputStream(teststream)
  val testlabels = ListBuffer.empty[DenseVector[Double]]

  println(testdataStream.readInt)
  val testcount = testdataStream.readInt()
  println(testcount)

  for (c <- 0 until testcount) {
    val label = testdataStream.readByte()
    testlabels += DenseVector.tabulate[Double](10)({ i => if (i == label) 1.0 else 0.0 })
  }
  println(testlabels.toList)
  testdataStream.close()

  val testimageStream = getClass.getResourceAsStream("/t10k-images-idx3-ubyte")
  val testimageDataStream  = new DataInputStream(testimageStream)
  val testimages = ListBuffer.empty[DenseVector[Double]]
  println(testimageDataStream.readInt())

  val testimageCount = testimageDataStream.readInt()
  val testheight = testimageDataStream.readInt()
  val testwidth = testimageDataStream.readInt()

  println(s"imageCount=${imageCount}, height=${testheight}, width=${testwidth}")

  for (c <- 0 until testimageCount) {
    val matrix = DenseMatrix.zeros[Int](testheight, testwidth)
    for (r <- 0 until testheight; c <- 0 until testwidth) {
      matrix(r, c) = testimageDataStream.readUnsignedByte()
    }
    //   println(matrix)
    testimages += DenseVector.tabulate(testheight * testwidth)({ i => matrix(i / testwidth, i % testheight) / 255.0 })
  }

  //println(images.toList)
  testimageDataStream.close


  val net = new network(List(784, 40, 25, 10))
  net.SGD(images.toList.zip(labels.toList), 30, 10, 3.0, Option(testimages.toList.zip(testlabels.toList)))
}
