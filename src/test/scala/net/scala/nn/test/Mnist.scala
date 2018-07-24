package net.scala.nn.test

import scala.collection.mutable.ListBuffer
import java.io.DataInputStream

import breeze.linalg.{DenseMatrix, DenseVector}
import net.scala.nn.{CrossEntropyCostNetwork, network}
import net.scala.nn.cnn._

/**
  * Created by wangmich on 06/26/2018.
  */
object Mnist extends App {

  private def loadLabels(file:String) = {
    val stream = getClass.getResourceAsStream(file)
    val dataStream  = new DataInputStream(stream)
    val labels = ListBuffer.empty[DenseVector[Double]]

    println(s"magic number: ${dataStream.readInt}")
    val count = dataStream.readInt() //skip  magic number
    println(s"label count=${count}")

    for (c <- 0 until count) {
      val label = dataStream.readByte()
      labels += DenseVector.tabulate[Double](10)({ i => if (i == label) 1.0 else 0.0 })
    }
    dataStream.close()
    labels.toList
  }
  private def loadImages(file:String) = {
    val imageStream = getClass.getResourceAsStream(file)
    val imageDataStream  = new DataInputStream(imageStream)
    val images = ListBuffer.empty[DenseVector[Double]]
    println(s"magic number: ${imageDataStream.readInt}") //skip magic number

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
    images.toList
  }
  def sigmoid (z:Double):Double = 1d/(1d+scala.math.exp(-z))
  val labels = loadLabels("/train-labels-idx1-ubyte")
  val images = loadImages("/train-images-idx3-ubyte")
  val testlabels = loadLabels("/t10k-labels-idx1-ubyte")
  val testimages = loadImages("/t10k-images-idx3-ubyte")


//  val net = new network(List(784, 40, 25, 10))
//  net.SGD(images.toList.zip(labels.toList), 30, 10, 3.0, Option(testimages.toList.zip(testlabels.toList)))
//  val (validationLabels, trainingLabels) = labels.splitAt(1000)
//  val (validationImages, trainingImages) = images.splitAt(1000)
//  val net1 = new network(List(784, 40, 25, 10)) with CrossEntropyCostNetwork
//  net1.SGD(images.toList.zip(labels.toList), 30, 10, 0.025, Option(testimages.toList.zip(testlabels.toList)))

  val net = CnnNetwork(List(new FullyConnectedLayer(784, 30, sigmoid, 0), new FullyConnectedLayer(30, 10, sigmoid, 0)).toArray)
  net.SGD(images.toList.zip(labels.toList), 30, 10, 3.0, Option(testimages.toList.zip(testlabels.toList)))
}
