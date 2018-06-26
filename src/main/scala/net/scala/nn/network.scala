package net.scala.nn

/**
  * Created by wangmich on 06/25/2018.
  */
abstract class network {

  def feedforward
  def SGD
  def update
  def backProp
  def eval


}
