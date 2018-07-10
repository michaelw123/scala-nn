package net.scala.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import breeze.stats.distributions.Rand

/**
  * Created by Michael Wang on 07/10/2018.
  */
object cnn {
  type NetworkVector = DenseVector[Double]
  type NetworkMatrix = DenseMatrix[Double]

  trait Layer {
    val in: Int
    val out: Int
    var b = DenseVector.rand(out, Rand.gaussian)
    var w = DenseMatrix.rand(in, out, Rand.gaussian)
  }

  trait Dropout {
    val dropout: Double
  }

  trait Activation {
    val actiovation: Double => Double
  }

  trait Convolution {
    val filter: filterShape
    val image: imageShape
  }

  trait Pool {
    def poolx: Int

    def pooly: Int
  }

  class filterShape {
    val nFilter = 0
    val nFeatures = 0
    val height = 0
    val width = 0
  }

  class imageShape {
    val batchSize = 0
    val nFeatures = 0
    val height = 0
    val width = 0
  }

  class FullyConnectedLayer(val in: Int, val out: Int, val actiovation: Double => Double, val dropout: Double) extends Layer with Activation with Dropout {

  }

  class SoftmaxLayer(val in: Int, val out: Int, val dropout: Double) extends Layer with Dropout {

  }

  class ConvPoolLayer(val in: Int, val out: Int, val dropout: Double, val poolx: Int, val pooly: Int, val filter: filterShape, val image: imageShape) extends Layer with Dropout with Convolution with Pool {

  }

  trait CnnNetwork extends network {
  }

  object CnnNetwork {
    def apply[L <: Layer](layers: Array[L]): network = new network(layers.map(l => l.in).toList)
  }

}