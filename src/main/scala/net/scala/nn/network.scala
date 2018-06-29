package net.scala.nn

/**
  * Created by Michael Wang on 06/25/2018.
  */
import breeze.linalg.{DenseMatrix, DenseVector, argmax, sum}
import breeze.stats.distributions.Rand
import breeze.generic.{MappingUFunc, UFunc}
import breeze.numerics.sigmoid

import scala.util.Random
object sigmoidPrime extends UFunc with MappingUFunc {

  implicit object sigmoidImplDouble extends Impl[Double, Double] {
    def apply(x: Double) = sigmoid(x) * (1d - sigmoid(x))
  }

}

 class network(private val sizes:List[Int]) {
   type NetworkVector = DenseVector[Double]
   type NetworkMatrix = DenseMatrix[Double]
   type TrainingSet = List[(NetworkVector, NetworkVector)]

   trait costFunction {
     def fn(a:NetworkVector, y:Double):Double
     def delta(z:NetworkVector, a:NetworkVector, y:NetworkVector):NetworkVector
   }

  protected var biases:List[NetworkVector] = this.sizes
    .drop(1)
    .map(n => DenseVector.rand(n, Rand.gaussian))

   protected var weights: List[NetworkMatrix] = this.sizes
     .drop(1)
     .zip(this.sizes.dropRight(1))
     .map({ case (r, c) => DenseMatrix.rand(r, c, Rand.gaussian) })

   protected val numLayers: Int = this.sizes.length

   def SGD(trainingData: TrainingSet, epochs: Int, miniBatchSize: Int, eta: Double, testData: Option[TrainingSet]): Unit = {

     var shuffledTrainingData = trainingData

     for (x <- 0 until epochs) {
       shuffledTrainingData = Random.shuffle(shuffledTrainingData)

       val miniBatches = shuffledTrainingData.grouped(miniBatchSize).toList

       miniBatches.foreach(b => updateMiniBatch(b, eta, 5.0, trainingData.size))

       testData match {
         case Some(d) => println(s"Epoch $x: ${evaluate(d)} / ${d.length}")
         case None => println(s"Epoch $x complete")
       }
     }
   }
   def updateMiniBatch(batch: TrainingSet, eta: Double, lmbta:Double = 0.0, n:Int = 0): Unit = {
     var newB = biases.map(b => DenseVector.zeros[Double](b.length))
     var newW = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

     batch.foreach({
       case (x, y) => {
         val (deltaB, deltaW) = backprop(x, y)

         newB = newB.zip(deltaB).map({ case (nb, db) => nb + db })
         newW = newW.zip(deltaW).map({ case (nw, dw) => nw + dw })

       }
     })
     biases = biases.zip(newB).map({ case (b, nb) => b - (eta / batch.length) * nb })
     weights = weights.zip(newW).map({ case (w, nw) => w - (eta / batch.length) * nw })
   }
   def backprop(x: NetworkVector, y: NetworkVector): (List[NetworkVector], List[NetworkMatrix]) = {
     val newB = biases.map(b => DenseVector.zeros[Double](b.length)).toArray
     val newW = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols)).toArray

     var activation = x

     var (zs, activations) = biases
       .zip(weights)
       .map({
         case (b, w) => {
           val z = w * activation + b
           activation = sigmoid(z)

           (z, activation)
         }
       })
       .unzip

     activations = x :: activations

     var delta = costDerivative(activations.last, y) *:* sigmoidPrime(zs.last)

     newB(newB.length - 1) = delta
     newW(newW.length - 1) = delta * activations(activations.length - 2).t

     for (l <- 2 until numLayers) {
       val z = zs(zs.length - l)
       val sp = sigmoidPrime(z)
       delta = (weights(weights.length - l + 1).t * delta) *:* sp

       newB(newB.length - l) = delta
       newW(newW.length - l) = delta * activations(activations.length - l - 1).t
     }

     (newB.toList, newW.toList)
   }
   def costDerivative(outputActivations: NetworkVector, y: NetworkVector): NetworkVector = {
     outputActivations - y
   }
   def evaluate(testData: TrainingSet): Int = {
     testData
       .map({ case (x, y) => (argmax(feedForward(x)), argmax(y)) })
       .count { case (x, y) => x == y }
   }
   def feedForward(activation: NetworkVector): NetworkVector = {
     var result = activation
     this.biases
       .zip(this.weights)
       .foreach({ case (b, w) => result = sigmoid(w * result + b) })

     result
   }
}
trait CrossEntropyCostNetwork extends network{
   object crossEntropyCost extends costFunction {
     def fn(a:NetworkVector, y:Double):Double = sum(a.map(x => -y * scala.math.log(x) - (1-y)*scala.math.log(1-x)))
     def delta(z:NetworkVector, a:NetworkVector, y:NetworkVector):NetworkVector = (a-y)
  }
  override def updateMiniBatch(batch: TrainingSet, eta: Double, lmbda:Double, n:Int): Unit = {
    var newB = biases.map(b => DenseVector.zeros[Double](b.length))
    var newW = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    batch.foreach({
      case (x, y) => {
        val (deltaB, deltaW) = backprop(x, y)
        newB = newB.zip(deltaB).map({ case (nb, db) => nb + db })
        newW = newW.zip(deltaW).map({ case (nw, dw) => nw + dw })

      }
    })
    biases = biases.zip(newB).map({ case (b, nb) => b - (eta / batch.length) * nb })
    weights = weights.zip(newW).map({ case (w, nw) => (1-eta*(lmbda/n)) *w - (eta / batch.length) * nw })
  }
  override def backprop(x: NetworkVector, y: NetworkVector): (List[NetworkVector], List[NetworkMatrix]) = {
    val newB = biases.map(b => DenseVector.zeros[Double](b.length)).toArray
    val newW = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols)).toArray

    var activation = x

    var (zs, activations) = biases
      .zip(weights)
      .map({
        case (b, w) => {
          val z = w * activation + b
          activation = sigmoid(z)
          (z, activation)
        }
      })
      .unzip
    activations = x :: activations

    //var delta = costDerivative(activations.last, y) *:* sigmoidPrime(zs.last)
    var delta = crossEntropyCost.delta(zs.last, activations.last, y)

    newB(newB.length - 1) = delta
    newW(newW.length - 1) = delta * activations(activations.length - 2).t

    for (l <- 2 until numLayers) {
      val z = zs(zs.length - l)
      val sp = sigmoidPrime(z)
      delta = (weights(weights.length - l + 1).t * delta) *:* sp

      newB(newB.length - l) = delta
      newW(newW.length - l) = delta * activations(activations.length - l - 1).t
    }

    (newB.toList, newW.toList)
  }
}
trait QuadraticCostNetwork extends network {
  object quadraticCost extends costFunction {
    def fn(a:NetworkVector, y:Double):Double = 0.5 * scala.math.pow(breeze.linalg.norm(a), 2)
    def delta(z:NetworkVector, a:NetworkVector, y:NetworkVector):NetworkVector = (a-y) * sigmoidPrime(z)
  }

}
trait trainingCostMonitoring extends network {

}
trait trainingAccuracyMonitoring extends network {

}
trait evaluationCostMonitoring extends network {

}
trait evaluationAccuracyMonitoring extends network {

}

