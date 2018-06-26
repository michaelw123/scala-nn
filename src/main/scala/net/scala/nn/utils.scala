package net.scala.nn

import breeze.numerics.sigmoid
/**
  * Created by Michael Wang on 06/25/2018.
  */
object utils {
  def sigmoid(z:Double) = 1.0/(1.0+scala.math.exp(-z))
}
