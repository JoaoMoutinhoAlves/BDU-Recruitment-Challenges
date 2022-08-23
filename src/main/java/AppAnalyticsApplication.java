import org.apache.spark.sql.DataFrameWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.time.LocalDateTime;
import java.util.Map;

import static org.apache.spark.sql.functions.*;

public class AppAnalyticsApplication {

    public static final String USER_REVIEWS_CSV = "googleplaystore_user_reviews.csv";
    public static final String APP_STORE_CSV = "googleplaystore.csv";
    private static final String RATING = "Rating";
    private static final String REVIEWS = "Reviews";
    private static final String APP = "App";
    private static final String CATEGORY = "Category";
    private static final String SENTIMENT_POLARITY = "Sentiment_Polarity";
    private static final double EUR2DOL_CONVERSION_RATE = 0.9;
    private static final String PRICE = "Price";
    private static final String SIZE = "Size";
    private static final String GENRE = "Genre";
    private static final String AVERAGE_SENTIMENT_POLARITY = "Average_Sentiment_Polarity";
    private static final String GENRES = "Genres";

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Play Store App Analytics")
                .master("local[2]")
                .config("spark.sql.warehouse.dir", "file:///C:/output/")
                .getOrCreate();

        // Part 1
        Dataset<Row> averageSentimentPolarityDS = createAverageSentimentPolarityDataset(spark.read()
                .option("header", true)
                .option("escape", "\"")
                .csv(USER_REVIEWS_CSV));

        // Part 2 - Apps with a Rating >= 4.0
        Dataset<Row> appStore = spark.read()
                .option("header", true)
                .option("escape", "\"")
                .csv(APP_STORE_CSV);
        Dataset<Row> bestApps = createBestAppsDataset(appStore);
        saveDatasetToFile(bestApps, "best_apps", "csv", false);

        // Part 3 - Remove duplicates
        Dataset<Row> duplicateFreeDataset = removeDuplicates(spark.read()
                .option("header", true)
                .option("escape", "\"")
                .csv(APP_STORE_CSV));

        // Part 4 - Join and save datasets
        Dataset<Row> cleanedDataset = joinDatasets(duplicateFreeDataset, averageSentimentPolarityDS);
        saveDatasetToFile(cleanedDataset, "googleplaystore_cleaned", "parquet", true);

        // Part 5 - Play Store Metrics
        Dataset<Row> appMetricsDataset = createMetricsDataset(duplicateFreeDataset, averageSentimentPolarityDS);
        saveDatasetToFile(appMetricsDataset, "googleplaystore_metrics", "parquet", true);

        spark.stop();
    }

    private static Dataset<Row> createAverageSentimentPolarityDataset(Dataset<Row> userReviews) {
        System.out.println("\n____________________________ Part 1 _________________________________\n");
        return userReviews.na().replace(SENTIMENT_POLARITY, Map.of("nan", "0"))
                .selectExpr(APP, "cast(Sentiment_Polarity as double) Sentiment_Polarity")
                .groupBy(APP)
                .avg(SENTIMENT_POLARITY).as(SENTIMENT_POLARITY);
    }

    private static Dataset<Row> createBestAppsDataset(Dataset<Row> appStore) {
        System.out.println("\n____________________________ Part 2 _________________________________\n");
        return appStore.filter(appStore.col(RATING).$greater$eq(4)).sort(desc(RATING));
    }

    private static Dataset<Row> removeDuplicates(Dataset<Row> appStore) {
        System.out.println("\n____________________________ Part 3 _________________________________\n");
        return appStore.orderBy(desc(REVIEWS)).groupBy(APP).agg(
                // Join all distinct grouped Categories
                collect_set(col(CATEGORY)).as("Categories"),
                // Selecting first occurrence since rows are already ordered by review count (desc), i.e. biggest
                // review count is the first occurrence
                first(col(RATING)),
                first(col(REVIEWS)),
                // Removing suffix from value before casting it to float
                regexp_extract(first(col(SIZE)), "(\\d+(.\\d+)).", 1)
                        .cast("float")
                        .as(SIZE),
                first(col("Installs")),
                first(col("Type")),
                // Removing dollar sign ($) from value before casting it to double
                regexp_extract(first(col(PRICE)), "$(.*)", 1)
                        .cast("double")
                        // Applying conversion rate to EURO
                        .$times(EUR2DOL_CONVERSION_RATE)
                        .as(PRICE),
                first(col("Content Rating").as("Content_Rating")),
                // Collecting all distinct Genres for each App
                flatten(collect_set(split(col(GENRES), ";"))).as(GENRES),
                // Casting date with custom format
                to_date(first(col("Last Updated")), "MMMM d, yyyy").as("Last_Updated"),
                first(col("Current Ver").as("Current_Version")),
                first(col("Android Ver").as("Minimum_Android_Version")));
    }

    private static Dataset<Row> joinDatasets(Dataset<Row> duplicateFreeDataset, Dataset<Row> averageSentimentPolarityDS) {
        System.out.println("\n____________________________ Part 4 _________________________________\n");
        return duplicateFreeDataset.join(averageSentimentPolarityDS, "App");
    }

    private static Dataset<Row> createMetricsDataset(Dataset<Row> cleanedDataset, Dataset<Row> averageSentimentPolarityDS) {
        System.out.println("\n____________________________ Part 5 _________________________________\n");
        return cleanedDataset.join(averageSentimentPolarityDS, APP)
                .select(
                        explode(col(GENRES)).as(GENRE),
                        col("first(Rating)").as(RATING),
                        col("Sentiment_Polarity.avg(Sentiment_Polarity)").as(AVERAGE_SENTIMENT_POLARITY))
                .groupBy(GENRE)
                .agg(
                        count(GENRE).as("Count"),
                        avg(RATING).as("Average_Rating"),
                        avg(AVERAGE_SENTIMENT_POLARITY).as(AVERAGE_SENTIMENT_POLARITY)
                );
    }

    private static void saveDatasetToFile(Dataset<Row> dataset, String filename, String format, boolean gzipCompression) {
        DataFrameWriter<Row> dfw = dataset.write();
        if (gzipCompression) {
            dfw = dfw.option("compression", "gzip");
        }
        dfw.option("header", true)
                .format(format)
                .save(filename + "_"
                        // Added timestamp to prevent loss of previously generated files
                        + LocalDateTime.now().toString().replace(":", "_")
                        + "."
                        + format);
    }
}
