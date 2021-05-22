package com.sparrowrecsys.offline.spark.model;
import java.net.URL;
import java.util.Iterator;
import java.util.List;

import jdk.nashorn.internal.codegen.CompilerConstants;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class JavaCollaborativeFiltering {
    static private JavaCollaborativeFiltering instance = null;
    private JavaCollaborativeFiltering (){
    }

    static public JavaCollaborativeFiltering getInstance(){
        if (instance == null){
            instance = new JavaCollaborativeFiltering();
        }
        return instance;
    }
    public static void main(String [] args){
        SparkConf sparkConf = new SparkConf().
                setMaster("local").
                setAppName("JavaCollaborativeFiltering").
                set("spark.submit.deployMode", "client");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        JavaCollaborativeFiltering collaborativeInstance = JavaCollaborativeFiltering.getInstance();
        URL filePath = collaborativeInstance.getClass().getResource("/webroot/sampledata/ratings.csv");
        SparkSession sparkSession = SparkSession.builder().config(sparkConf).getOrCreate();
        Dataset<Row> ratingSamples = sparkSession.read().format("csv").option("header", true).load(filePath.getPath()).
                withColumn("UserIdInt", new Column("userId").cast("int")).
                withColumn("MovieIdInt", new Column("movieId").cast("int")).
                withColumn("ratingFloat", new Column("rating").cast("float"));

        double[] ratio_array = new double[2];
        ratio_array[0] = 0.8;
        ratio_array[1] = 0.2;
        Dataset<Row>[] split_array = ratingSamples.randomSplit(ratio_array);
        Dataset<Row> train_data_set = split_array[0];
        Dataset<Row> test_data_set = split_array[1];
        ALS als = new ALS();
        als.setMaxIter(5);
        als.setRegParam(0.01);
        als.setUserCol("UserIdInt");
        als.setItemCol("MovieIdInt");
        als.setRatingCol("ratingFloat");

        ALSModel model = als.fit(train_data_set);
        model.setColdStartStrategy("drop");
        Dataset<Row> test_result = model.transform(test_data_set);
        model.itemFactors().show(10, false);
        model.userFactors().show(10, false);

        RegressionEvaluator evaluator = new RegressionEvaluator();
        evaluator.setMetricName("rmse");
        evaluator.setLabelCol("ratingFloat");
        evaluator.setPredictionCol("prediction");

        double evaluator_result = evaluator.evaluate(test_result);
        System.out.println("evaluator_result_rmse" + Double.toString(evaluator_result));

        Dataset<Row> allUserRecommend = model.recommendForAllUsers(10);
        Dataset<Row> allItemRecommend = model.recommendForAllItems(10);

        Dataset<Row> threeUsers = ratingSamples.select(als.getUserCol()).distinct().limit(3);
        Dataset<Row> threeUserRecResult = model.recommendForUserSubset(threeUsers, 10);

        Dataset<Row> threeMovies = ratingSamples.select(als.getItemCol()).distinct().limit(3);
        Dataset<Row> threeMoviesRecResult = model.recommendForItemSubset(threeMovies, 10);

        allUserRecommend.show(false);
        allItemRecommend.show(false);
        threeUserRecResult.show(false);
        threeMoviesRecResult.show(false);

        double[] regParamArray = new double[2];
        regParamArray[0] = 0.01;
        regParamArray[1] = 0.1;
        ParamMap[] paramGridBuilder = new ParamGridBuilder().addGrid(als.regParam(), regParamArray).build();

        CrossValidator crossValidator = new CrossValidator();
        crossValidator.setEstimator(als);
        crossValidator.setEvaluator(evaluator);
        crossValidator.setEstimatorParamMaps(paramGridBuilder);
        crossValidator.setNumFolds(3);

        CrossValidatorModel cv_model = crossValidator.fit(test_data_set);
        double[] avgMetrics = cv_model.avgMetrics();
        sparkSession.stop();
    }
}
