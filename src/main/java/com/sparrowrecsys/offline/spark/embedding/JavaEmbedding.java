package com.sparrowrecsys.offline.spark.embedding;

import java.lang.reflect.Array;
import java.net.URL;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import javafx.util.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.functions;
import org.apache.spark.api.java.*;
import org.apache.spark.sql.types.DataTypes;
import scala.Tuple2;
import scala.collection.Seq;
import scala.collection.JavaConverters;
import scala.collection.JavaConversions;
import org.apache.spark.mllib.feature.Word2Vec;

class ExtractMoviesSequence implements UDF1<Seq<Row>, Seq<String>> {
    @Override
    public Seq<String> call(Seq<Row> rowSeq){
        List<Row> rowList = JavaConversions.seqAsJavaList(rowSeq);
        List<String> resultList = rowList.
                stream().
                map(row -> new Pair<String, Integer>(row.getString(0), row.getInt(1))).
                sorted(new Comparator<Pair<String, Integer>>() {
                    @Override
                    public int compare(Pair<String, Integer> o1, Pair<String, Integer> o2) {
                        return o1.getValue() - o2.getValue();
                    }
                }).map(onePair -> onePair.getKey()).collect(Collectors.toList());
        return JavaConverters.asScalaIteratorConverter(resultList.iterator()).asScala().toSeq();
    }
}
public class JavaEmbedding {
    static private JavaEmbedding instance = null;
    private JavaEmbedding(){

    }
    static public JavaEmbedding getInstance(){
        if (instance == null){
            instance = new JavaEmbedding();
        }
        return instance;
    }

    static public JavaRDD<String[]> getEmbeddingSequence(Dataset<Row> ratingDataSet){

        ratingDataSet.printSchema();
        Dataset<Row> userMovieIdsDataSet = ratingDataSet.
                where(new Column("rating").gt(3.5)).
                groupBy(new Column("userId")).
                agg(
                        functions.callUDF("extractMoviesSequence",
                                functions.collect_list(
                                        functions.struct(
                                                new Column("movieId"),
                                                new Column("timestamp").cast("int")
                                        )
                                )
                        ).as("movieIds")
                ).
                withColumn("movieIdStr", functions.array_join(new Column("movieIds"), " "));
        userMovieIdsDataSet.printSchema();
        userMovieIdsDataSet.show(10, false);
        userMovieIdsDataSet.select(new Column("userId"), new Column("movieIds")).show(10, false);
        return userMovieIdsDataSet.select("movieIdStr").javaRDD().map(row -> row.getString(0).split(" "));
    }

    static public Word2Vec Embedding(JavaRDD<List<String>> movieIdsRdd, int maxSize, int vecSize, int numIter){

        Word2Vec word2VecModel = new Word2Vec();
        word2VecModel.setVectorSize(vecSize).setNumIterations(numIter).setWindowSize(maxSize);
        Word2VecModel fitWord2VecModel = word2VecModel.fit(movieIdsRdd);
        Tuple2<String, Object>[] synonyms = fitWord2VecModel.findSynonyms("158", 20);
        for (Tuple2<String, Object> synonym : synonyms){
            System.out.println(synonym._1.toString() +" "+synonym._2.toString());
        }
        return word2VecModel;
    }

    static public void main(String[] arg){

        JavaEmbedding instance = JavaEmbedding.getInstance();
        URL filePathUrl = instance.getClass().getResource("/webroot/sampledata/ratings.csv");

        SparkConf sparkConf = new SparkConf().
                setMaster("local").
                setAppName("JavaEmbedding").
                set("spark.submit.deployMode", "client");
        SparkSession sparkSession = SparkSession.builder().config(sparkConf).getOrCreate();

        sparkSession.udf().register("extractMoviesSequence", new ExtractMoviesSequence(), DataTypes.createArrayType(DataTypes.StringType));
        Dataset<Row> ratingDataSet = sparkSession.read().format("csv").option("header", true).load(filePathUrl.getPath());
        JavaRDD<String[]> ratingSequence = JavaEmbedding.getEmbeddingSequence(ratingDataSet);
        JavaRDD<List<String>> ratingListRdd = ratingSequence.map(new Function<String[], List<String>>() {
            @Override
            public List<String> call(String[] strings) throws Exception {
                List<String> stringList = new ArrayList<>();
                for (String oneStr : strings) {
                    stringList.add(oneStr);
                }
                return stringList;
            }
        });

        JavaEmbedding.Embedding(ratingListRdd, 5, 10, 5);

    }
}
