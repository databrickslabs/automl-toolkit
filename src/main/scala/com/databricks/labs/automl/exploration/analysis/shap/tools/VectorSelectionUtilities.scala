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

  // TODO Convert this to sample with replacement
  def buildSelectionSet(indexUnderTest: Int,
                        partitionVectorSize: Int,
                        vectorCountToSelect: Int,
                        randomSeed: Random): mutable.LinkedHashSet[Int] = {

    val selectionPayload = mutable.LinkedHashSet.empty[Int]

    assert (vectorCountToSelect <= partitionVectorSize, "vectorCountToSelect must be less than partitionVectorSize")

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

  def mutateVectors(mutationVectors: Map[Int, Map[Int, Double]],
                    referenceVector: Map[Int, Double],
                    referenceIndex: Int,
                    vectorIndexListing: List[Int],
                    partitionVectorSize: Int,
                    selectedVectorsForMutation: List[Int],
                    seed: Random): Array[MutatedVectors] = {

    selectedVectorsForMutation
      .map(k => {

        val randomVectorsForSample =
          selectRandomVectorIndeces(vectorIndexListing, referenceIndex)

        val vectorIncludeReference = vectorIndexListing
          .map {
            case x if randomVectorsForSample.contains(x) =>
              mutationVectors(k)(x)
            case x if !randomVectorsForSample.contains(x) => referenceVector(x)
          }
          .toArray[Double]

        val vectorReplaceReference = vectorIndexListing
          .map {
            case x if randomVectorsForSample.contains(x) =>
              mutationVectors(k)(x)
            case x if x == referenceIndex =>
              mutationVectors(k)(x)
            case x
                if !randomVectorsForSample.contains(x) && x != referenceIndex =>
              referenceVector(x)
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
