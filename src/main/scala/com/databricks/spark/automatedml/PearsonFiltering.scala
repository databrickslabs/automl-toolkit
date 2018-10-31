package com.databricks.spark.automatedml

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.ml.stat.ChiSquareTest
import scala.collection.mutable.ListBuffer

//TODO: finish this for dynamic and manual filtering of feature fields based on pearson relevance scores.

class PearsonFiltering {

  case class PearsonPayload(fieldName: String, pvalue: Double, degreesFreedom: Int, pearsonStat: Double)

  def buildChiSq(df: DataFrame, fields: Array[String]): List[PearsonPayload] = {
    val reportBuffer = new ListBuffer[PearsonPayload]

    val chi = ChiSquareTest.test(df, "features", "label").head
    val pvalues = chi.getAs[Vector](0).toArray
    val degreesFreedom = chi.getSeq[Int](1).toArray
    val pearsonStat = chi.getAs[Vector](2).toArray

    for(i <- fields.indices){
      reportBuffer += PearsonPayload(fields(i), pvalues(i), degreesFreedom(i), pearsonStat(i))
    }
    reportBuffer.result
  }

  def manualFilterChiSq(statPayload: List[PearsonPayload], filterStat: String, filterDirection: String, fields: Array[String], filterValue: Double): List[String] = {
    val fieldRestriction = new ListBuffer[String]
    filterDirection match {
      case "greaterthan" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach {f =>
            f.setAccessible(true)
            if(f.getName == filterStat) if(f.get(x).asInstanceOf[Double] >= filterValue) fieldRestriction += x.fieldName else None else None
          }
        })
      case "lessthan" =>
        statPayload.foreach(x => {
          x.getClass.getDeclaredFields foreach {f =>
            f.setAccessible(true)
            if(f.getName == filterStat) if(f.get(x).asInstanceOf[Double] <= filterValue) fieldRestriction += x.fieldName else None else None
          }
        })
      case _ => throw new UnsupportedOperationException("Can't support that") //TODO: make a real error message here
    }
    fieldRestriction.result
  }

  def quantileGenerator(percentile: Double, stat: String, pearsonResults: List[PearsonPayload]): Double = {

    assert(percentile < 1 & percentile > 0, "Percentile Value must be between 0 and 1.")
    val statBuffer = new ListBuffer[Double]
    pearsonResults.foreach(x => {
      x.getClass.getDeclaredFields foreach {f=>
        f.setAccessible(true)
        if(f.getName == stat) statBuffer += f.get(x).asInstanceOf[Double]
      }
    })

    val statSorted = statBuffer.result.sortWith(_<_)
    if(statSorted.size % 2 == 1) statSorted((statSorted.size * percentile).toInt)
    else {
      val splitLoc = math.floor(statSorted.size * percentile).toInt
      val splitCheck = if(splitLoc < 1) 1 else splitLoc.toInt
      val(high, low) = statSorted.splitAt(splitCheck)
      (high.last + low.head) / 2
    }

  }


}
