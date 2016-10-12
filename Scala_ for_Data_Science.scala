// MlLlib's Local Vectors
// 1. linear algebra operations provided by Breeze and Jblas
// 2. Scala imports scala.collection.immutable.Vector by default 
// 3. You have to import org.apache.spark.mllib.linalg.{Vector, Vectors} to use MLlib Vectors 
// 4. Indices are 0-based integers and values are Doubles
// 5. local vectors are stored on a single machine.
// 6. MLlib's vectors can be either dense or sparse

// Dense Vector
// 1. A dense vector is backed by a double array containing its values 
// 2. Easily created from an array of doubles

import org.apache.spark.mllib.linalg.{Vector,Vectors}
Vectors.dense(44.0, 0.0, 55.0)

// Sparse Vector
// 1. A sparse vector is backed by two arrays: an integer array representing the indexes; a double array containing the corresponding values
// 2. It's a binary-search implementation of a vector. 
// 3. Can be created by specifying the indices and values for non-zero entries as either: two separate arrays; a sequence of tuples. 

Vectors.sparse(3, Array(0,2), Array(44.0, 55.0))
Vectors.sparse(3, Seq((0, 44.0), (2, 55.0)))

// Labeled Points
// 1. The association of a vector, either dense or sparse, with a corresponding label/response.
// 2. Used in supervised machine learning algorithm. 
// 3. Labels are stored as doubles so they can be used in both regression and classification problems
// 4. In classification problems, labels must be: 0(negative) or 1(positive) for binary classification; class indices starting from zero (0,1,2,...) for multiclass.

import org.apache.spark.mllib.regression.LabeledPoint
LabeledPoint(1.0, Vectors.dense(44.0, 0.0, 55.0))
LabeledPoint(0.0, Vectors.sparse(3, Array(0,2), Array(44.0, 55.0)))

// Local Matrices
// 1. Natural extension of vectors
// 2. Row and column indices are 0-based integer and values are doubles
// 3. Local matrices are stored on a single machine
// 4. MLlib's matrices can be either dense or sparse
// 5. Matrices are filled in column major order.

// Dense Matrices
// 1. A "reshaped" dense vector
// 2. First two arguments specify dimensions of the matrix
// 3. Entries are stored in a single double array.

import org.apache.spark.mllib.linalg.{Matrix, Matrices}
Matrices.dense(3, 2, Array(1,3,5,2,4,6))

// Sparse Matrices in Spark: Compressed Sparse Column(CSC) format
val m = Matrices.sparse(5, 4, Array(0,0,1,2,2), Array(1,3), Array(34,55))

// Distributed Matrices
 
// 1. Distributed matrices are where Spark starts to deliver significant values.
// 2. They are stored in one or more RDDs.
// 3. Three types have been implemented: RowMatrix; IndexedRowMatrix; CoordinateMatrix
// 4. Conversions may require a expensive global shuffle.

// RowMatrix

// 1. The most basic type of distributed matrix.
// 2. It has no meaningful row indices, being only a collection of feature vectors.
// 3. Backed by a RDD of its rows, where each row is a local vector. 
// 4. Assumes the number of columns is small enough to be stored in a local vector.
// 5. Can be easily created from a instance of RDD[Vector]

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val rows: RDD[Vector] = sc.parallelize(Array(Vectors.dense(1.0, 2.0), Vectors.dense(4.0, 5.0), Vectors.dense(7.0, 8.0)))
val mat: RowMatrix = new RowMatrix(rows)
val m = mat.numRows()
val n = mat.numCols()

// IndexedRowMatrix 

// 1. Similar to a RowMatrix
// 2. But is has meaningful row indices, which can be used for identifying rows and executing joins
// 3. Backed by an RDD of indexed rows, where each row is a tuple containing an index (long-typed) and a local vector.
// 4. Easily created from an instance of RDD[IndexedRow]
// 5. can be converted to a RowMatrix by calling toRowMatrix()

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
val rows: RDD[IndexedMatrix] = sc.parallelize(Array(
		IndexedRow(0, Vectors.dense(1.0, 2.0)),
		IndexedRow(1, Vectors.dense(4.0, 5.0)),
		IndexedRow(2, Vectors.dense(7.0, 8.0))))

val idxMat: IndexedRowMatrix = new IndexedRowMatrix(rows)

// CoordinateMatrix

// 1. Should be used only when both dimensions are huge and matrix is very sparse
// 2. Backed by a RDD of matrix entries, where each entry is a tuple (i:Long, j:Long, value:Double) where: i is the row index, j is the column index, and value is the entry value.
// 3. Can be easily created from an instance of RDD[MatrixEntry]
// 4. Can be converted to an IndexRowMatrix with sparse rows by calling toIndexedRowMatrix()

import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix

val entries: RDD[MatrixEntry] = sc.parallelize(Array(
		MatrixEntry(0, 0, 9.0),
		MatrixEntry(1, 1, 8.0),
		MatrixEntry(2, 1, 6.0)))

val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Summary Statistics

// 1. Column summary statistics for an instance of RDD[Vector] are available through the colStats() function in statistics.
// 2. It returns an instance of MultivariateStatisticalSummary, which contains column-wise results for: min, max, mean, variance, numNonzeros, normL1, normL2
// 3. Count returns the total count of elements

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary

val observations: RDD[Vector] = sc.parallelize(Array(
		Vectors.dense(1.0, 2.0),
		Vectors.dense(4.0, 5.0),
		Vectors.dense(7.0, 8.0)))

val summary: MultivariateStatisticalSummary = Statistics.colStats(observatios)
summary.mean
summary.variance
summary.numNonzeros
summary.normL1
summary.normL2

// Correlations

// 1. Pairwise correlations among series is available through the corr() function in Statistics
// 2. Correlation methods supported: Pearson (defaults) and Spearman (used for rank variables)
// 3. Input supported: two RDD[Double]s, returning a single Double value; one RDD[Vector], returning a correlation Matrix

val x: RDD[Double] = sc.parallelize(Array(2.0, 9.0, -7.0))
val y: RDD[Double] = sc.parallelize(Array(1.0, 3.0, 5.0))
val correlation: Double = Statistics.corr(x,y, "pearson")

val data: RDD[Vector] = sc.parallelize(Array(
		Vectors.dense(2.0, 9.0, -7.0),
		Vectors.dense(1.0, -3.0, 5.0),
		Vectors.dense(4.0, 0.0, -5.0)))
val correlMatrix: Matrix = Statistics.corr(data, "pearson")

// Random Data Generation

// 1. RandomRDDs: generate either random double RDDs or vector RDDs.
// 2. Supported distributions: uniform, normal, lognormal, Poisson, exponential, and gamma
// 3. Useful for randomized algorithms, prototyping, and performance testing.

import org.apache.spark.mllib.random.RandomRDDs._
val million = poissonRDD(sc, mean=1.0, size=1000000L, numPartitions=10)
million.mean 
million.variance

val data = normalVectorRDD(sc, numRows=10000L, numCols=3, numPartitions=10)
val stats: MultivariateStatisticalSummary = Statistics.colStats(data)
stats.mean
stats.variance

// Sampling

// 1. Can be performed on any RDD
// 2. Returns a sampled subset of a RDD
// 3. Sampling with or without replacement
// 4. Fraction: without replacement, expecting size of the sample as fraction of RDD' size; with replacement, expected number of times each element is chosen.
// 5. can be used on boostrapping procedures

import org.apache.spark.mllib.linalg.{Vector, Vectors}
val elements: RDD[Vector] = sc.parallelize(Array(
		Vectors.dense(4.0, 7.0, 13.0),
		Vectors.dense(-2.0, 8.0, 4.0),
		Vectors.dense(3.0, -11.0, 19.0)))

elements.sample(withReplacement=false, fraction=0.5, seed=10L).collect()

// Random Split

// 1. Can be performed on any RDD
// 2. Return an array of RDDs
// 3. Weights for the split will be normalized if they do not add up to 1
// 4. Useful for splitting a dataset into training, test, and validations sets.

val data = sc.parallelize(1 to 1000000)
val splits = data.randomSplit(Array(0.6, 0.2, 0.2), seed=13L)
val traning = splits(0)
val test = splits(1)
val validation = splits(2)
splits.map(_.count())

// Stratified Sampling

// 1. Can be performed on RDDs of key-value pairs. 
// 2. Think of keys as labels and values as an specific attribute
// 3. Two supported methods defined in PairRDDFunctions: sampleByKey: requires only one pass over the data and provides an expected sample size; sampleByKeyExact: provides the exact sampling size with 99.99% confidence, but requires significantly more resources.

import org.apache.spark.mllib.linalg.distributed.IndexedRow

val rows: RDD[IndexedRow] = sc.parallelize(Array(
		IndexedRow(0, Vectors.dense(1.0, 2.0)),
		IndexedRow(1, Vectors.dense(4.0, 5.0)),
		IndexedRow(1, Vectors.dense(7.0, 8.0))))

val fractions: Map[Long, Double] = Map(0L -> 1.0, 1L -> 0.5)
val approxSample = rows.map{ case IndexedRow(index, vect) => (index, vect)}
	.sampleByKey(withReplacement=false, fractions, 9L)

approxSample.collect()

// Hypothesis Testing

// 1. Used to determine whether a result is significant statistically, that is, whether it occurs by chance or not.
// 2. Supported tests: Pearson's chi-squared test for goodness of fit; Pearson's chi-squared test for independence; Kolmogorov-Smirnov test for equality of distribution.
// 3. Inputs of type RDD[LabeledPoint] are also supported, enabling feature selection. 

// Pearson's Chi-Squared Test for Goodness-of-Fit

// 1. Determines whether an observed frequency distribution differs from a given distribution or not. 
// 2. Requires an input of type Vector containing the frequencies of the events
// 3. It runs against a uniform distribution, if a second vector to test against is not supplied.
// 4. Available as chiSqTest() function in statistics

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.mllib.stat.Statistics

val vec: Vector = Vectors.dense(0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05)
val goodnessofFitTestResult = Statistics.chiSqTest(vec)

// Pearson's Chi-Square Test for Independence 

// 1. Determines whether unpaired observations on two variables are independent of each other. 
// 2. Requires an input of type Matrix, representing a contingency table, or RDD[LabeledPoint]
// 3. Available as chiSqTest() function in Statistics

val mat: Matrix = Matrices.dense(3, 2, Array(13.0, 47.0, 40.0, 80.0, 11.0, 9.0))
val independenceTestResult = Statistics.chiSqTest(mat)

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib. stat.test.ChiSqTestResult

val obs: RDD[LabledPoint] = sc.parallelize(Array(
		LabeledPoint(0, Vectors.dense(1.0, 2.0)),
		LabeledPoint(0, Vectors.dense(0.5, 1.5)),
		LabeledPoint(1, Vectors.dense(1.0, 8.0))))

val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)

// Kolmogorov-Smirnov Test

// 1. Determines whether or not two probability distributions are equal
// 2. one sample , two sided test
// 3. supported distributions to test against: normal distribution (distName="norm"); customized cumulative density function (CDF)
// 4. Available as kolmogorovSmirnovTest() function in Statistics

import org.apache.spark.mllib.random.RandomRDDs.normalRDD 
val data: RDD[Double] = normalRDD(sc, size=100, numPartitions=1, seed= 13L)
val testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 0, 1)

// Kernel Density Estimation 

// 1. Computes an estimate of the probability density function of a random variable, evaluated at a given set of points
// 2. Does not require assumptions about the particular distribution that the observed samples are drawn from
// 3. Requires an RDD of samples
// 4. Available as estimate() function in KernelDensity
// 5. In Spark, only Gaussian kernel is supported

import org.apache.spark.mllib.stat.KernelDensity
val data: RDD[Double] = normalRDD(sc, size=1000, numPartitions=1, seed=17L)
val kd = new KernelDensity().setSample(data).setBandWidth(0.1)
val densities = kd.estimate(Array(-1.5, -1, -0.5, 0, 0.5, 1, 1.5))

// Statistics, Random Data, Sampling on DataFrames

// Summary Statistics for DataFrames

// 1. Column summary statistics for DataFrames are available through DataFrame's describe() method
// 2. It returns another DataFrame, which contains columnwise result for: min, max, mean, stddev, count. 
// 3. column summary statistics can also be computed through DataFrame's groupBy() and agg() methods, but stddev is not supported.
// 4. It also returns another DataFrame with the result.

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

case class Record(desc: String, value1: Int, value2: Double)

val recDF = sc.parallelize(Array(
		Record("first", 1, 3.7),
		Record("second", -2, 2.1),
		Record("third", 6, 0.7))).toDF()
val recStats = recDF.describe()
recStats.show()

recStats.filter("summary='stddev'").first()
recStats.filter("summary='stddev'").first().toSeq.toArray
recStats.filter("summary='stddev'").first().toSeq.toArray.drop(1).map(_.toString.toDouble)
recStats.select("value1").map(s => s(0).toString.toDouble).collect()

// Another Example
recDF.groupBy().agg(Map("value1"->"min", "value1" -> "max")) // only return max
recDF.groupBy().agg(Map("value1" -> "min", "value2" -> "min"))

import org.apache.spark.sql.functions._
val recStatsGroup = recDF.groupBy().agg(min("value1"), min("value2"))
recStatsGroup.columns
recStatsGroup.first().toSeq.toArray.map(_.toString.toDouble)

// More Statistics on DataFrames

// 1. More statistics are available through the stats method in a DataFrame
// 2. It returns a DataFrameStatsFunctions object, which has the following methods: corr() - computes Pearson correlation between two columns; cov() computes sample covariance between two columns; crosstab() - computes a pairwise frequency table of the given columns; freqItems() - finds frequent items for columns, possibly with false positives.

val recDFStat = recDF.stat
recDFStat.corr("value1", "value2")
recDFStat.cov("value1", "value2")
recDFSata.freqItems(Seq("value1"), 0.3).show()

// Sampling on DataFrames

// 1. Can be performed on any DataFrame
// 2. Returns a sampled subset of a DataFrame
// 3. Sampling with or without replacement
// 4. fraction: expected fraction of rows to generate
// 5. can be used on boostrapping procedures

val df = sqlContext.createDataFrame(Seq(
	(1,10), (1,20), (2,10), (2,20), (2,30),(3,20),(3,30))).toDF("key","value")
val dfSampled = df.sample(withReplacement=false, fractions=0.3, seed=1L)

// Random Split on DataFrames

// 1. Can be performed on any DataFrame
// 2. Returns an array of DataFrames
// 3. Weights for the split will be normalized if they do not add up to 1
// 4. Useful for splitting a dataset into training, test, and validation set

val dfSplit = df.randomSplit(weights=Array(0.3, 0.7), seed=1L)
dfSplit(0)
dfSplit(1)

// Stratified Sampling on DataFrame

// 1. Can be performed on any DataFrame
// 2. Any column may work as key
// 3. without replacement
// 4. fraction: specified by key
// 5. Available as sampleBy function in DataFrameStatFunctions

val dfStrat = df.Stat.sampleBy(col="key", fractions=Map(1-> 0.7, 2->0.7, 3 ->0.7), seed=11L) 

// Random Data Generation

// 1. SQL functions to generate columns filled with random values
// 2. Two supported distributions: uniform and normal
// 3. Useful for randomized, prototyping, and performance testing.

import org.apache.spark.sql.functions.{rand, randn}
val df = sqlContext.range(0,10)
df.select("id").withColumn("uniform", random(10L)).withColumn("normal", random(10L))

// Handing Missing Data and Inputing Values

// DataFrame NA Functions

// 1. The na method of DataFrames provides functionality for working with missing data
// 2. Returns an instance of DataFrameNAFunctions
// 3. The following methods are available: drop - for dropping rows containing NaN or null values; fill - for replacing NaN or null values; replace - for replacing values matching specified keys

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import org.apache.spark.sql.fuctions._
val df = sqlContext
	.range(0, 10)
	.select("id")
	.withColumn("uniform", random(10L))
	.withColumn("normal", random(10L))

val halfToNaN = udf[Double, Double](x=> if(x>0.5) Double.NaN else x)
val oneToNaN = udf[Double, Double](x => if(x>1.0) Double.NaN else x)

val dfnan = df
	.withColumn("nanUniform", halfToNaN(df("uniform")))
	.withColumn("nanNormal", oneToNaN(df("normal")))
	.drop("uniform")
	.withColumnRenamed("nanUniform","uniform")
	.drop("normal")
	.withColumnRenamed("nanNormal", "normal")
dfnan.show()

// DataFrame NA Functions - drop

// 1. drop is used to dropping rows containing NaN or null values according to a criteria. 
// 2. Several implementation available: drop(minNonNulls, cols); drop(minNonNulls); drop(how, cols); drop(cols); drop(how); drop()
// 3. cols is an Array or Seq of column names
// 4. how is equal any or all

dfnan.na.drop(minNonNulls=3).show()
dfnan.na.drop("all", Array("uniform", "normal")).show()
dfnan.na.drop("any", Array("uniform", "normal")).show()

// DataFrame NA Functions - fill

// 1. fill is used for replacing NaN or null values according to some criteria
// 2. Several implementation available: fill(valueMap); fill(value,cols); fill(value)

dfnan.na.fill(0.0).show()
val uniformMean = dfnan.filter("uniform <>'NaN'").groupBy().agg(mean("uniform")).first()(0)
dfnan.na.fill(Map("uniform" -> uniformMean)).show(5)

val dfCols = dfnan.columns.drop() 
val dfMeans = dfnan.na.drop().groupBy().agg(mean("uniform"), mean("normal")).first().toSeq
val meanMap = (dfCols.zip(dfMeans())).toMap
dfnan.na.fill(meanMap).show(5)

// DataFrame NA Functions - replace

// 1. Replace is used for replacing values matching specific keys
// 2. cols argument may be a single column name or an array
// 3. replacement argument is a map: key is the value to be matched; value is the replacement value itself.

dfnan.na.replace("uniform", Map(Double.NaN -> 0.0)).show()

// Duplicates

// 1. dropDuplicates is a DataFrame method
// 2. Used to remove duplicate rows
// 3. May specify a subset of columns to check for duplicates 

import sqlContext.implicits._
val dfDuplicates = df.unionAll(sc.parallelize(Seq((10,1,1),(11,1,1))).toDF())
val dfCols = dfnan.columns.drop(1)
dfDuplicates.dropDuplicates(dfCols).show()

// Transformers and Estimators

// Transformer

// 1. Algorithm: transform one DataFrame into another DataFrame
// 2. Implements a method transform(), which converts one DataFrame into another, generally by appending one or more columns.
// 3. Input and output columns set with setInputCol and setOutputCol methods
// 4. Examples: read one or more columns and map them into a new column of feature vectors; read a column containing feature vectors and make a prediction for each other.

import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{Tokenizer, RegexTokenizer}

val sqlContext = new SQLContext(sc)
val sentenceDataFrame = sqlContext.createDataFrame(Seq(
	(0, "Hi I want to learn Spark"),
	(1, "I wish Java could use case classes"),
	(2, "Logistic regression, models, are, neat"))).toDF("label","sentence") 
val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val tokenized = tokenizer.transform(sentenceDataFrame)

// Estimator

// 1. Algorithm: fit on a DataFrame to produce a Transformer
// 2. Implements a method fit() , which accepts a DataFrame and produces a model, which is a Transformer
// 3. Example: Logistic Regression It is a learning algorithm and therefore an Estimator. By calling the method fit() to train the logistic regression. A model is returned. 

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}

val training = sqlContext.createDataFrame(Seq(
	(1.0, Vectors.dense(0.0, 1.0, 0.1)),
	(0.0, Vectors.dense(2.0, 1.0, -1.0)),
	(0.0, Vectors.dense(2.0, 1.0, 1.0)),
	(1.0, Vectors.dense(0.0, 1.2, -0.5)))).toDF("label","features")

val lr = new LogisticRegression()
	.setMaxIter(10)
	.setRegParam(0.01)

val model = lr.fit(training)
model.transform(training).show()

// Parameters

// 1. Transformers and estimators use a uniform API for specifying parameters
// 2. A paramMap is a set of (parameter, value) pairs
// 3. Parameters are specific to a give instance
// 4. There are two main ways to pass parameters to an algorithm: setting parameters for an instance using an appropriate method, for instance: setMaxIter(10); passing a ParamMap to fit() or transform(), for instance, ParamMap(lr1.MaxIter-> 10, lr2.MaxIter->20)

import org.apache.spark,ml.param.ParamMap

val paramMap = ParamMap(lr.maxIter -> 20, lr.regParam -> 0.01)
val model = lr.fit(trainingm paramMap)
model.transform(training).show()

// Vector Assembler

// 1. Transformer that combines a given list of columns into a single vector column
// 2. Useful for combining raw features and features generated by other transformer into a single feature vector.
// 3. Accept the following input column types: all numeric types; boolean; vector
 import org.apache.spark.ml.feature.VectorAssembler
 import org.apache.spark.sql.functions._

 val dfRandom = sqlContext
 	.range(0,10)
 	.select("id")
 	.withColumn("uniform", rand(10L))
 	.withColumn("normal1", randn(10L))
 	.withColumn("normal2", randn(11L))

 val assembler = new VectorAssembler()
 	.setInputCols(Array("uniform", "normal1", "normal2"))
 	.setOutputCol("features")

 val dfVec = assembler.transform(dfRandom)
 dfVec.select("Id", "features").show()

 // Data Normalization 

 // Normalizer

 // 1. A Transformer which transforms a dataset of Vector rows, normalizing each Vector to have unit norm.
 // 2. Takes a parameter P, which specifies the p-norm used for normalization (p=2 by default).
 // 3. Standardized input data and improve the behavior of learning algorithms

import org.apache.spark.ml.feature.Normalizer

val scaler1 = new Normalizer()
	.setInutCol("features")
	.setOutputCol("scaledFeat")
	.setP(1.0)

scaler1.transform(dfVec.select("Id", "features")).show(5)

// StandardScaler

// 1. A model which can be fit on a dataset to produce a StandardScalerModel
// 2. A Transformer which transforms a dataset of Vector rows, normalizing each feature to have unit standard deviation and/or zero mean
// 3. Takes two parameters: withStd: scales the data to unit standard deviation (default: true); withMean: centers the data with mean before scaling (default:false)
// 4. It builds a dense output, sparse inputs will raise an exception.
// 5. If the standard deviation of a feature is zero, it returns 0.0 in the Vector for that feature.

import org.apache.spark.ml.feature.StandardScaler

val scaler2 = new StandardScaler()
	.setInputCol("features")
	.setOutputCol("scaledFeat")
	.setWithStd(true)
	.setWithMean(true)

val scaler2Model = scaler2.fit(dfVec.select("Id","features"))
scaler2Model.transform(dfVec.select("Id","features")).show(5)

// MinMax Scaler

// 1. A model which can be fit on a dataset to produce a MinMaxScalerModel
// 2. A Transformer which transform a dataset of Vector rows, rescaling each feature to a specific range ( often [0,1])
// 3. Takes two parameters: min - lower bound after transformation, shared by all features (default: 0.0); max - upper bound after transformation, shared by all features (default: 1.0)
// 4. Since zero values are likely to be transformed to non-zero values, Sparse inputs map result in dense outputs.

import org.apache.spark.ml.feature.MinMacScaler

val scaler3 = new MinMaxScaler()
	.setInputCol("features")
	.setOutputCol("scaledFeat")
	.setMin(-1.0)
	.setMax(1.0)

val scaler3Model = scaler3.fit(dfVec.select("Id","features"))
scaler3Model.transform(dfVec.select("Idi","features")).show(5)

// Identifying Outlier

// Mahalanobis Distance

// 1. Multiple-dimensional generalization of measuring how many standard deviations a point is away from the mean.
// 2. Measured along each Principle Component axis
// 3. Unitless and scale-invariant
// 4. Takes into account the correlations of the dataset
// 5. Used to detect outliers

import org.apache.spark.mllib.linalg.{Vector, Vectors}  
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val dfRandom = sqlContext
	.range(0,10)
	.select("Id")
	.withColumn("uniform", rand(10L))
	.withColumn("normal1", randn(10L))
	.withColumn("normal2", randn(11L)) 

val assembler = new VectorAssembler()
	.setInputCols(Array("uniform","normal1", "normal2"))
	.setOutputCol("features")

val dfVec = assembler.transform(dfRandom)
dfVec.select("Id","features").show()

val dfOutlier = dfVec
	.select("Id","features")
	.unionAll(sqlContext.createDataFrame(Seq(10, Vectors.dense(3,3,3))))
dfOutlier.sort(dfOutlier("Id").desc).show(5)

val scaler = new StandardScaler()
	.setInputCol("features")
	.setOutputCol("scaledFeat")
	.setWithMean(true)
	.setWithStd(true)

val scalerModel = scaler.fit(dfOutlier.select("Id","features"))

val dfScaled = scalerModel.transform(dfOutlier).select("Id","scaledFeat") 
dfScaled.sort(df("Id").desc).show(3)

import org.apache.spark.mllib.stat.Statistics
import breeze.linalg._

val rddVec = dfScaled.select("scaledFeat").rdd.map(_.(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
val colCov = Statistics.corr(rddVec)
val invColCovB = inv(new DenseMatrix(3,3,colCov.toArray))

val mahalanobis = udf[Doube, org.apache.spark.mllib.linalg.Vector]{
	v => val vB = DenseVector(v.toArray());
		 vB.t * invColCovB * vB
}
val dfMahalanobis = dfScaled.withColumn("mahalanobis", mahalanobis(dfScaled("scaledFeat")))

// Removing Outliers
dfMahalanobis.sort(dfMahalanobis("mahalanobis").desc).show(2)
val ids = dfMahalanobis.select("Id","mahalanobis").sort(dfMahalanobis("mahalanobis").desc).drop("mahalanobis").collect()
val idOutliers = ids.map(_.(0).asInstanceOf[Long]).slice(0,2)
dfOutlier.filter("id not in (0,2)")

// Feature Vectors

// 1. MLlib: the models in MLlib are designed to work with RDD[LabeledPoint] objects with associate labels with feature vectors
// 2. Spark.ML: The models in spark.ml are designed to work with DataFrames. A basic spark.ml DataFrame will (by default) have two columns. a label column (default name: label); a features column (default name: features)
// 3. The output of your ETL process might be a DataFrame with various columns. For example, you might want to try to predict churn based on number of sessions, revenue, and recency

case class Customer(churn:Int, sessions: Int, revenue: Double, recency: Int)

val customers = {sqlContext.SparkContext.parallelize(
	Customer(1, 20, 61.24, 103) ::
	Customer(1, 8, 80.64, 23) :: 
	Customer(0, 4, 100.94, 42) ::
	Customer(0, 8, 99.48, 26) ::
	Customer(1, 17, 120.56, 47) :: Nil).toDF()}

val assembler = new VectorAssembler()
	.setInputCols(Array("sessions", "revenue", "recency"))
	.setOutputCol("features")

val dfWithFeatures = assembler.transform(customers)

// VectorSlicers

import org.apache.spark.ml.feature.VectorSlicer

val slicer = new VectorSlicer()
	.setInputCol("features")
	.setOutputCol("some_features")

slicer.setIndices(Array(0,1)).transform(dfWithFeatures)

// Categorical Features

// 1. Spark's classifiers and regresors only work with numerical features, String features must be converted to numbers a StringIndexer
// 2. This keeps spark's internals simpler and more efficient
// 3. There is a little cost in transforming categorical features to numbers, and then back to strings.

// StringIndexer

val df = sqlContext.createDataFrame(Seq(
	(0, "US"),
	(1, "UK"),
	(2, "FR"),
	(3, "US"),
	(4, "US"),
	(5, "FR"))).toDF("id","nationality")

import org.apache.spark.ml.feature.StringIndexer

val indexer = new StringIndexer()
	.setInputCol("nationality")
	.setOutputCol("nIndex")

val indexed = indexer.fit(df).transform(df)

// Reversing the Mapping

// 1. The classifiers in MLlib and spark.ml will predict numeric values that correspond to the index values
// 2. IndexToString is what you'll need to transform these numbers back into your original labels

import org.apache.spark.ml.feature.IndexToString

val converter = new IndexToString()
	.setInputCol("predictedIndex")
	.setOutputCol("predictedNationality")

val predictions = indexed.selectExpr("nIndex as predictedIndex")
converter.transform(predictions)

// One Hot Encoding

// The oneHotEncoder creates a sparse vector column, with each dimension of this vector of Booleans representing one of the possible values of the original feature.

import org.apache.spark.ml.feature.OneHotEncoder

val encoder = new OneHotEncoder()
	.setInputCol("nIndex")
	.setOutputCol("nVector")

val encoded = encoder.transform(indexed)

import org.apache.spark.mllib.linalg._

encoded.foreach{c => val dv = c.getAs[SparseVector]("nVector").toDense;
					 println(s"${c(0)} ${c(1)} ${dv}")}

// The dropLast option to keep all features

val encoder = new OneHotEncoder()
	.setInputCol("nIndex")
	.setOutputCol("nVector")
	.setDropLast(false)
val encoded = encoder.transform(indexed)
val toDense = udf[DenseVector, SparseVector](_.toDense)
encoded.withColumn( "denseVector", toDense(encoded("nVector")))

// Explode, UDFm and Pivot

case class Sales(id: Int, account: String, year: String, commission: Int, sales_reps: Seq[String])

val sales = sqlContext.createDataFrame(Seq(
	Sales(1, "Acme", "2013", 1000, Seq("Jim", "Tom")), 
	Sales(2, "Lucas", "2013", 1100, Seq("Fred", "Ann")),
	Sales(3, "Acme", "2014", 2800, Seq("Jim")))).toDF

import org.apache.spark.sql.functions._
val len: (Seq[String] => Int) => (arg: Seq[String]) => {arg.length}
val column_len = udf(len)

val exploded = sales.select($"id", $"account", $"year", ($"commission" / column_len($"sales_reps")), explode($"sales_reps").as("sales_rep"))

// Pivot() with groupBy()
exploded.groupBy($"sales_rep").pivot("year").agg(sum("share")).orderBy("sales_rep").collect

// PCA in Feature Engineering
val crimes = sqlContext
	.read
	.format("com.databricks.spark.csv")
	.option("delimiter", ",")
	.option("header", "true")
	.option("inferSchema", "true")
	.load("~/crime.csv")

// convert from DataFrame to RowMatrix
val assembler = new VectorAssembler
	.setInputCols(crime.columns)
	.setOutputCol("features")

val featuresDF = assembler.transform(crimes).select("features")
val rddOfRows = featuresDF.rdd

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val rddOfVectors = rddOfRows.map(row => row.get(0).asInstanceOf[Vector])
val mat = new RowMatrix(rddOfVectors)
val pc: Matrix = mat.computePrincipleComponents(10)

// 1. Principal Components are stored in a local dense matrix

// RFormula

// 1. An RFormula object produces a vector column of features and a double column of labels
// 2. String input columns will be one-hot encoded, and numerical columns will be cast to doubles.

// Define a model with RFormula interface
import org.apache.spark.ml.feature.RFormula

val formula = new RFormula()
	.setFormula("ViolentCrimePerPop ~ householdSize + PctLess9thGrade + PctWage")
	.setFeaturesCol("features")
	.setLabelCol("label")
// Run the model and show the output. This is a regression model, so "label" is a bit misleading
val output = formula.fit(crimes).transform(crimes)
output.select("features","label").show(3)

// Decision Trees

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

val dtc = new DecisionTreeClassifier()
	.setLabelCol("indexedLabel")
	.setFeaturesCol("indexedFeatures")

import org.apache.spark.ml.Pipeline 
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}

val data = MLUtils.loadLibSVMFile(sc, "/sample_libsvm_data.csv").toDF()

// Creating the Tree Model
val labelIndexer = new StringIndexer()
	.setInputCol("label")
	.setOutputCol("IndexedLabel")
	.fit(data)

val labelConverter = new IndexToString()
	.setInputCol("prediction")
	.setOutputCol("predictedLabel")
	.setLabels(labelIndexer.labels)

val featureIndexer = new VectorIndexer()
	.setInputCol("features")
	.setOutputCol("IndexedFeatures")
	.setMaxCategories(4)
	.fit(data)

val pipelineClass = new pipeline()
	.setStages(Array(labelIndexer, featureIndexer, dtc, labelConverter))

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
val modelClassifier = pipelineClass.fit(traningData)
val treeModel = modelClassifier.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model: \n" + treeModel.toDebugString)
val predictedClass = modelClassifier.transform(testData)

// Decision Regressor
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel

val dtR = new DecisionTreeRegressor()
	.setLabelCol("label")
	.setFeaturesCol("indexedFeatures")

val pipelineReg = new Pipeline()
	.setStages(Array(featureIndexer, dtR))

val modelRegressor = pipelineReg.fit(trainingData)
val treeModel = modelRegressor.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned Regression tree model: \n" + treeModel.toDebugString)
val predictionsReg = modelRegressor.transform(testData)

// Random Forests

// RF Classification
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel

val rfc = new RandomForestClassifier()
	.setInputCol("IndexedLabel")
	.setOutputCol("IndexedFeatures")
	.setNumTrees(3)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
val pipelineRFC = new Pipleline()
	.setStages(Array(labelIndexer, featureIndexer, rfc, labelConverter))
val modelRFC = pipelineRFC.fit(trainingData)
val predictionsRFC = modelRFC.transfom(testData)
predictionsRFC.select("predictedLabel", "label","features").show(5)

val rfModelC = modelRFC.stages(2).asInstanceOf[RandomForestClassificationModel]
rfModelC.featureImportances
println("Learned classification random forest model: \n" + rfModelC.toDebugString)

// RF for Regression
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.RandomForestRegressionModel

val rfR = new RandomForestRegressor()
	.setLabelCol("label")
	.setFeaturesCol("indexedFeatures")

val pipelineRFR = new Pipeline()
	.setStages(Array(featureIndexer, rfR))
val modelRFR = pipelineRFR.fit(trainingData)
val predictionsRFR = modelRFR.transform(testData)
predictionsRFR.select("prediction","label","features").show(5)

// Gradient Boosting Trees
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.classification.GBTClassificationModel

val gbtC = new GBTClassifier()
	.setLabelCol("indexedLabel")
	.setFeaturesCol("indexedFeatures")
	.setMaxIter(10)

val pipelineGBTC = new Pipeline()
	.setStages(Array(labelIndexer, featureIndexer, gbtC, labelConverter))

val modelGBTC = pipelineGBTC.fit(trainingData)
val predictionsGBTC = modelGBTC.transform(testData)
predictionsGBTC.select("predictedLabel","label","features").show(3)
val gbtModelC = modelGBTC.stages(2).asInstaneOf[GBTClassificationModel]
println("learned classification GBT model: \n" + gbtModelC.toDebugString)

// GBT Regression
import org.apache.spark.ml.classification.GBTRegressor
import org.apache.spark.ml.classification.GBTRegressionModel

val gbtR = new GBTRegressor()
	.setLabelCol("label")
	.setFeaturesCol("indexedFeatures")
	.setMaxIter(10)

val pipelineGBTR = new Pipeline()
	.setStages(Array(featureIndexer, gbtR))

val modelGBTR = pipelineGBTR.fit(trainingData)
val predictionsGBTR = modelGBTR.transform(testData)

// Linear Methods
// Example of Logistic Regression

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.BinaryLogiticRegressionSummary

val logr = new LogisticRegression()
	.setMaxIter(10)
	.setRegParam(0.3)
	.setElasticNetParam(0.8)

val logrModel = logr.fit(trainingData{}
println(s"Weights: ${logrModel.coefficients}, Intercept: ${logrModel.intercept}")
val trainingSummaryLR = logrModel.summary
val objectiveHistoryLR = trainingSummaryLR.objectiveHistory

// Linear Least Squares
import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
	.setMaxIter(10)
	.setRegParam(0.3)
	.setElasticNetParam(0.8)

val lrModel = lr.fit(traningData)
println(s"Weights: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")
val trainingSummaryLLR = lrModel.summary
println(s"Number of Iterations: ${trainingSummaryLLR.totalIterations}")
println(s"Objective History: ${trainingSummaryLLS.objectiveHistory.toList}")
trainingSummaryLLR.residuals.show(3)

// Evaluation

// BinaryClassificationEvaluator
// 1. Evaluator for binary classification
// 2. Expects two input columns: rawPrediction and label
// 3. Supported metric: areaUnderROC

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val predictionsLogR = logrModel.transform(testData)
val evaluator = new BinaryClassificationEvaluator()
	.setLabelCol("label")
	.setRawPredictionCol("rawPrediction")
	.setMetricName("areaUnderROC")

// This is a close-to-perfect model
val roc = evaluator.evaluate(predictionsLogR)

// MulticlassClassificationEvaluator
// 1. Expects two input columns: prediction and label 
// 2. Supported metrics: F1 (default), Precision, Recall, WeightedPrecision, WeightedRecall

// Reusing RF Classification:
import org.apache.spark.ml.evaluation.MulticlassClassificationEvalutor
val evaluator = new MulticlassClassficationEvaluator()
	.setLabelCol("indexedLabel")
	.setPredictionCol("prediction")
	.setMetricName("precision")

val accuracy = evaluator.evaluate(predictionsRFC)
println("Test Error = " + (1.0 - accuracy))

// RegressionEvaluator
// 1. Evaluator for regression
// 2. Expected two input columns: prediction and label
// 3. Supported metrics: rmse - root mean squared error(default); mse: mean squared error; r2: the coefficient of determination; mae: mean absolute error.

// Reusing RF Regression Example:
import org.apache.spark.ml.evulation.RegressionEvaluator

val evaluator = new RegressionEvaluator()
	.setLabelCol("label")
	.setPredictionCol("prediction")
	.setMetricName("rmse")

val rmse = evaluator.evaluate(predictionRFR)
println("Root Mean Squared Error(RMSE) = " + rmse)

// BinaryLogisticRegressionSummary

// 1. LogisticRegressionTrainingSummary accessible through summary attribute of a LogisticRegressionModel
// 2. Summarizes the model over the training set.
// 3. Can be casted as BinaryLogisticRegressionSummary

val traningSummaryLR = logrModel.summary
val binarySummary = traningSummaryLR.asInstanceOf[BinaryLogisticRegressionSummary]
println(binarySummary.areaUnderROC)
val fMeasure = binarySummary.fMeasureByThreshold
fMeasure.show(3)

val maxFMeasure = fMeasure.agg("F-Measure" -> "max").head().getDouble(0)
val bestThreshold = fMeasure.wehre($"F-Measure" === "maxFMeasure").select("threshold").head().getDouble(0)
binarySummary.pr.show(3)
binarySummary.precisionByThreshold.show(3)
binarySummary.recallByThreshold.show(3)
binarySummary.roc.show(3)

// LinearRegressionTraningSummary

// 1. Accessible through summary attribute of LinearRegressionModel
// 2. Summarizes the model over the training set

// Reusing Linear Regression Example
val traningSummaryLLS = lrModel.summary
trainingSummaryLLS.explainedVariance
trainingSummaryLLS.meanAbsoluteError
trainingSummaryLLS.meanSquaredError
trainingSummaryLLS.r2
trainingSummaryLLS.residuals.show(3)
trainingSummaryLLS.rootMeanSqauredError

v
