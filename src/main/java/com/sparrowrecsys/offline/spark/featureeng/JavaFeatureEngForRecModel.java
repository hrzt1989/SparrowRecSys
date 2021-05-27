package com.sparrowrecsys.offline.spark.featureeng;
import java.awt.peer.CanvasPeer;
import java.net.URL;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.*;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.Seq;
import scala.collection.JavaConverters;
import scala.collection.JavaConversions;
import org.apache.spark.sql.expressions.Window;
import scala.collection.convert.Decorators;
/**
addMovieFeatures
 |-- movieId: string (nullable = true)
 |-- userId: string (nullable = true)
 |-- rating: string (nullable = true)
 |-- timestamp: string (nullable = true)
 |-- label: integer (nullable = false)
 |-- genres: string (nullable = true)
 |-- releaseYear: integer (nullable = false)
 |-- movieGenre1: string (nullable = true)
 |-- movieGenre2: string (nullable = true)
 |-- movieGenre3: string (nullable = true)
 |-- movieRatingCount: long (nullable = true)
 |-- movieAvgRating: string (nullable = true)
 |-- movieRatingStddev: string (nullable = true)
*/

/**
 addUserFeatures
 |-- movieId: string (nullable = true)
 |-- userId: string (nullable = true)
 |-- rating: string (nullable = true)
 |-- timestamp: string (nullable = true)
 |-- label: integer (nullable = false)
 |-- releaseYear: integer (nullable = false)
 |-- movieGenre1: string (nullable = true)
 |-- movieGenre2: string (nullable = true)
 |-- movieGenre3: string (nullable = true)
 |-- movieRatingCount: long (nullable = false)
 |-- movieAvgRating: string (nullable = true)
 |-- movieRatingStddev: string (nullable = true)
 |-- userRatedMovie1: string (nullable = true)
 |-- userRatedMovie2: string (nullable = true)
 |-- userRatedMovie3: string (nullable = true)
 |-- userRatedMovie4: string (nullable = true)
 |-- userRatedMovie5: string (nullable = true)
 |-- userRatingCount: long (nullable = false)
 |-- userAvgReleaseYear: integer (nullable = false)
 |-- userReleaseYearStddev: string (nullable = true)
 |-- userAvgRating: string (nullable = true)
 |-- userRatingStddev: string (nullable = true)
 |-- userGenre1: string (nullable = true)
 |-- userGenre2: string (nullable = true)
 |-- userGenre3: string (nullable = true)
 |-- userGenre4: string (nullable = true)
 |-- userGenre5: string (nullable = true)
 */

class ExtractYear implements UDF1<String, Integer>{
    @Override
    public Integer call(String title){
        if (title != null && title.length() >= 6){
            return Integer.parseInt(title.substring(title.length() - 5, title.length() - 1));
        }
        return 1990;
    }
}

class ExtractTitle implements UDF1<String, String> {

    @Override
    public String call(String title){
        return title.substring(0, title.length() - 6).trim();
    }
}
class ExtractGenres implements UDF1<Seq<String>, Seq<String>>{
    @Override
    public Seq<String> call(Seq<String> stringSeq){
        Map<String, Integer> genreCountMap = new HashMap<>();
        List<String> javaList = JavaConversions.seqAsJavaList(stringSeq);
        for (String genres : javaList){
            String[] genreArray = genres.split("\\|");
            for (String genre : genreArray){
                Integer genreCount = genreCountMap.getOrDefault(genre, 0) + 1;
                genreCountMap.put(genre, genreCount);

            }
        }
        LinkedList<Map.Entry<String, Integer>> genreCountPairList = new LinkedList(genreCountMap.entrySet());
        genreCountPairList.sort(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue() - o2.getValue();
            }
        });
        LinkedList<String> result = new LinkedList<String>();
        for (Map.Entry<String, Integer> pair : genreCountPairList){
            String genre = pair.getKey();
            result.add(genre);
        }
        return JavaConverters.asScalaIteratorConverter(result.iterator()).asScala().toSeq();
    }
}

public class JavaFeatureEngForRecModel {
    static private JavaFeatureEngForRecModel instance = null;
    static public JavaFeatureEngForRecModel getInstance(){
        if (instance == null){
            instance = new JavaFeatureEngForRecModel();
        }
        return instance;
    }
    private JavaFeatureEngForRecModel(){
    }
    static public Dataset<Row> addLabel(Dataset<Row> ratingDataSet){
        long sampleCount = ratingDataSet.count();

        ratingDataSet.groupBy(new Column("rating")).
                count().
                withColumn("ratingCount", new Column("count")).
                orderBy(new Column("rating")).
                withColumn("percentage", new Column("count").divide(sampleCount)).
                show(10, false);

        Dataset<Row> ratingDatasetWithLabel = ratingDataSet.
                withColumn("label",
                        functions.when(new Column("rating").gt(new Float(3.5)), 1).otherwise(0)
                );
        ratingDatasetWithLabel.schema();
        ratingDatasetWithLabel.show(10, false);
        return ratingDatasetWithLabel;
    }

    static public Dataset<Row> addMoviesFeature(Dataset<Row> moviesDataSet, Dataset<Row> ratingWithLabelDataSet){

        List<String> joinIdList = new LinkedList<>();
        joinIdList.add("movieId");
        Dataset<Row> ratingMoviesJoinDataSet = ratingWithLabelDataSet.join(
                moviesDataSet,
                JavaConverters.asScalaIteratorConverter(joinIdList.iterator()).asScala().toSeq(),
                "left"
        );

        Dataset <Row> ratingMoviesJoinDataSet0 = ratingMoviesJoinDataSet.
                withColumn("releaseYear",
                        functions.callUDF("extractYear", new Column("title"))).
                withColumn("title",
                        functions.callUDF("extractTitle", new Column("title")));

        Dataset<Row> ratingMoviesJoinDataSet1 = ratingMoviesJoinDataSet0.
                withColumn("movieGenre1",
                        functions.split(new Column("genres"), "\\|").getItem(0)).
                withColumn("movieGenre2",
                        functions.split(new Column("genres"), "\\|").getItem(1)).
                withColumn("movieGenre3",
                        functions.split(new Column("genres"), "\\|").getItem(2));

        Dataset<Row> movieRatingFeatures = ratingMoviesJoinDataSet1.groupBy(new Column("movieId")).
                agg(
                        functions.lit(1).as("movieRatingCount"),
                        functions.format_number(functions.avg(new Column("rating")), 2).as("movieAvgRating"),
                        functions.stddev(new Column("rating")).as("movieRatingStddev")
                ).na().fill(0).
                withColumn("movieRatingStddev", functions.format_number(new Column("movieRatingStddev"), 2));

        Dataset<Row> result = ratingMoviesJoinDataSet1.join(
                movieRatingFeatures,
                JavaConverters.asScalaIteratorConverter(joinIdList.iterator()).asScala().toSeq(),
                "left").drop("title");
        return result;

    }
    static public Dataset<Row> addUserFeature(Dataset<Row> movieFeatureDataSet){

        Dataset<Row> userPositiveHistoryDataSet = movieFeatureDataSet.
                withColumn("userPositiveHistory",
                        functions.collect_list(
                                functions.when(
                                        new Column("label").notEqual(1),
                                        new Column("movieId")
                                ).otherwise(null)).
                                over(
                                        Window.partitionBy(new Column("userId")).
                                                orderBy(new Column("timestamp")).rowsBetween(-100, -1))
                );

        Dataset<Row> result = userPositiveHistoryDataSet.withColumn("userPositiveHistory",
                functions.reverse(new Column("userPositiveHistory"))).
                withColumn("userRatedMovie1", new Column("userPositiveHistory").getItem(0)).
                withColumn("userRatedMovie2", new Column("userPositiveHistory").getItem(1)).
                withColumn("userRatedMovie3", new Column("userPositiveHistory").getItem(2)).
                withColumn("userRatedMovie4", new Column("userPositiveHistory").getItem(3)).
                withColumn("userRatedMovie5", new Column("userPositiveHistory").getItem(5)).
                withColumn("ratingCount",
                        functions.count(functions.lit(1)).
                                over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                )

                ).
                withColumn("userAvgReleaseYear",
                        functions.avg(new Column("releaseYear")).
                                over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                )
                ).
                withColumn("userReleaseYearStddev",
                        functions.stddev(new Column("releaseYear")).
                                over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                )
                ).
                withColumn("userAvgRating",
                        functions.format_number(functions.avg(new Column("rating")).
                                over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                ), 2
                        )
                ).
                withColumn("userRatingStddev",
                        functions.stddev(new Column("rating")).
                                over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                )
                ).
                withColumn("userGenres",
                        functions.callUDF("extractGenres",
                                functions.collect_list(new Column("genres")).over(
                                        Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)
                                ))
                ).na().fill(0).
                withColumn("userRatingStddev", functions.format_number(new Column("userRatingStddev"), 2)).
                withColumn("userReleaseYearStddev", functions.format_number(new Column("userReleaseYearStddev"), 2)).
                withColumn("userGenre1", new Column("userGenres").getItem(0)).
                withColumn("userGenre2", new Column("userGenres").getItem(1)).
                withColumn("userGenre3", new Column("userGenres").getItem(2)).
                withColumn("userGenre4", new Column("userGenres").getItem(3)).
                withColumn("userGenre5", new Column("userGenres").getItem(4)).
                drop("userGenres", "userPositiveHistory", "genres").filter(new Column("ratingCount").gt(1));

        return result;
    }
    static public void main(String[] arg){
        SparkConf sparkConf = new SparkConf().
                setMaster("local").
                setAppName("JavaFeatureEngForRecModel").
                set("spark.submit.deployMode", "client");

        JavaFeatureEngForRecModel modelInstance = JavaFeatureEngForRecModel.getInstance();

        URL ratingUrl = modelInstance.getClass().getResource("/webroot/sampledata/ratings.csv");
        URL moviesUrl = modelInstance.getClass().getResource("/webroot/sampledata/movies.csv");

        SparkSession sparkSession = SparkSession.builder().config(sparkConf).getOrCreate();

        sparkSession.udf().register("extractYear", new ExtractYear(), DataTypes.IntegerType);
        sparkSession.udf().register("extractTitle", new ExtractTitle(), DataTypes.StringType);
        sparkSession.udf().register("extractGenres", new ExtractGenres(), DataTypes.createArrayType(DataTypes.StringType));


        Dataset<Row> ratingDataSet = sparkSession.read().format("csv").option("header", true).load(ratingUrl.getPath());
        Dataset<Row> moviesDataSet = sparkSession.read().format("csv").option("header", true).load(moviesUrl.getPath());

        ratingDataSet.show(10, false);
        Dataset<Row> ratingWithLabelSet = JavaFeatureEngForRecModel.addLabel(ratingDataSet);
        Dataset<Row> moviesFeatureDataSet = JavaFeatureEngForRecModel.addMoviesFeature(moviesDataSet, ratingWithLabelSet);
        Dataset<Row> userFeatureDataSet = JavaFeatureEngForRecModel.addUserFeature(moviesFeatureDataSet);
        System.out.println("userFeatureDataSet");
        userFeatureDataSet.schema();
        userFeatureDataSet.show(10, false);
        sparkSession.stop();
        
    }
}
