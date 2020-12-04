package com.databricks.labs.automl.exploration.analysis.shap.tools

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row

import scala.collection.{immutable, mutable}
import scala.util.Random

/**
  * Utility objects for performing SHAP calculations
  */
private[analysis] object VectorSelectionUtilities extends Serializable {

  def extractFeatureCollection(
    partitionData: Array[Row],
    featureCol: String
  ): immutable.Map[Int, immutable.Map[Int, Double]] = {

    partitionData
      .map(
        _.getAs[org.apache.spark.ml.linalg.Vector](featureCol).toDense.toArray.zipWithIndex
          .map(x => (x._2, x._1))
          .toMap
      )
      .zipWithIndex
      .map(x => (x._2, x._1))
      .toMap

  }

  @scala.annotation.tailrec
  def selectRandomVector(indexUnderTest: Int,
                         partitionVectorSize: Int,
                         random: Random): Int = {
    val selection = random.nextInt(partitionVectorSize)
    selection match {
      case x if x == indexUnderTest =>
        selectRandomVector(indexUnderTest, partitionVectorSize, random)
      case _ => selection
    }
  }

  def buildSelectionSet(indexUnderTest: Int,
                        partitionVectorSize: Int,
                        vectorCountToSelect: Int,
                        randomSeed: Random): mutable.MutableList[Int] = {
    val selectionPayload = mutable.MutableList.empty[Int]

    do {
      selectionPayload += selectRandomVector(
        indexUnderTest,
        partitionVectorSize,
        randomSeed
      )
    } while (selectionPayload.size < vectorCountToSelect)

    selectionPayload
  }

  def selectRandomVectorIndeces(vectorIndexListing: List[Int],
                                vectorToTest: Int): List[Int] = {

    val randomVectorIndeces = Random.shuffle(vectorIndexListing)
    val indexOfTestingVector = randomVectorIndeces.indexOf(vectorToTest)
    randomVectorIndeces.take(indexOfTestingVector)
  }

  def extractVectors(
    partitionPayload: Map[Int, Array[Map[Int, Double]]],
    indecesToExtract: mutable.LinkedHashSet[Int]
  ): List[Array[Map[Int, Double]]] = {

    indecesToExtract.map(x => partitionPayload(x)).toList

  }

  def mutateVectors(vectorData: Map[Int, Map[Int, Double]],
                    referenceFeatureVector: Map[Int, Double],
                    referenceFeatureIndex: Int,
                    featureIndices: List[Int],
                    partitionVectorSize: Int,
                    sampleVectorIndices: List[Int],
                    seed: Random): Array[MutatedVectors] = {

    sampleVectorIndices
      .map(k => {

        val sampleFeatureIndicesToMutate =
          selectRandomVectorIndeces(featureIndices, referenceFeatureIndex)

        val vectorIncludeReference = featureIndices
          .map {
            case x if sampleFeatureIndicesToMutate.contains(x) =>
              vectorData(k)(x)
            case x if !sampleFeatureIndicesToMutate.contains(x) => referenceFeatureVector(x)
          }
          .toArray[Double]

        val vectorReplaceReference = featureIndices
          .map {
            case x if sampleFeatureIndicesToMutate.contains(x) =>
              vectorData(k)(x)
            case x if x == referenceFeatureIndex =>
              vectorData(k)(x)
            case x
                if !sampleFeatureIndicesToMutate.contains(x) && x != referenceFeatureIndex =>
              referenceFeatureVector(x)
          }
          .toArray[Double]

        MutatedVectors(
          Vectors.dense(vectorIncludeReference),
          Vectors.dense(vectorReplaceReference)
        )

      })
      .toArray

  }

}
