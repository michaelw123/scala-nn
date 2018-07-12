package net.scala.nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import breeze.stats.distributions.Rand

import scala.util.Random

/**
  * Created by Michael Wang on 07/10/2018.
  */
object cnn {
  type NetworkVector = DenseVector[Double]
  type NetworkMatrix = DenseMatrix[Double]
  type TrainingSet = List[(NetworkVector, NetworkVector)]

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

  class CnnNetwork {
    var layers:List[Layer]=Nil
    def SGD(trainingData: TrainingSet, epochs: Int, miniBatchSize: Int, eta: Double, testData: Option[TrainingSet]):Unit = {
      var shuffledTrainingData = trainingData

      for (x <- 0 until epochs) {
        shuffledTrainingData = Random.shuffle(shuffledTrainingData)
        val miniBatches = shuffledTrainingData.grouped(miniBatchSize).toList
        miniBatches.foreach(b => updateMiniBatch(b, eta, 5.0, trainingData.size))
//        val acc = accuracy(shuffledTrainingData)
//        testData match {
//          case Some(d) => println(s"Epoch $x: ${evaluate(d)} / ${d.length}")
//          case None => println(s"Epoch $x complete")
//        }
//        println(s"accuracy: ${acc}:/ ${shuffledTrainingData.length}")
      }
    }
    def updateMiniBatch(batch: TrainingSet, eta: Double, lmbta:Double = 0.0, n:Int = 0): Unit = {
      var newB = layers.map(l => DenseVector.zeros[Double](l.b.length))
      var newW = layers.map(l => DenseMatrix.zeros[Double](l.w.rows, l.w.cols))

      batch.foreach({
        case (x, y) => {
          val (deltaB, deltaW) = backprop(x, y)
          newB = newB.zip(deltaB).map({ case (nb, db) => nb + db })
          newW = newW.zip(deltaW).map({ case (nw, dw) => nw + dw })
        }
      })
      layers.zip(newB).zip(newW).map({case ((l, nb), nw) =>
        l.b = l.b - (eta / batch.length) * nb
        l.w = l.w - (eta / batch.length) * nw
      })
     }
    def backprop(x: NetworkVector, y: NetworkVector): (List[NetworkVector], List[NetworkMatrix]) = {
      var newB = layers.map(l => DenseVector.zeros[Double](l.b.length))
      var newW = layers.map(l => DenseMatrix.zeros[Double](l.w.rows, l.w.cols))

      var activation = x

      var (zs, activations) = layers.map(l => {
        val z = l.w * activation + l.b
        activation = sigmoid(z)
        (z, activation)
      }).unzip
      activations = x :: activations

      var delta = costDerivative(activations.last, y) *:* sigmoidPrime(zs.last)

      layers.last.b = delta
      layers.last.w = delta * activations(activations.length - 2).t

      val (b, w) = layers.tail.zip(newB.tail).zip(activations.reverse.tail).map( {case ((l, nb), a)  =>
        val z = zs.last
        val sp = sigmoidPrime(z)
        delta = l.w.t * delta *:* sp
        (delta, delta * a.t)
      }).unzip
      ((newB.head :: b).toList, (newW.head :: w).toList)
    }
    def costDerivative(outputActivations: NetworkVector, y: NetworkVector): NetworkVector = {
      outputActivations - y
    }
  }

  object CnnNetwork {
    def apply[L <: Layer](l: Array[L]): CnnNetwork = new CnnNetwork{ layers = l.toList }
  }

}