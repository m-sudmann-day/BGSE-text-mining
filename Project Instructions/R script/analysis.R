
if (!require(RMySQL)) { install.packages("RMySQL") }
if (!require(randomForest)) { install.packages("randomForest") }
if (!require(ggplot2)) { install.packages("ggplot2") }
if (!require(Metrics)) { install.packages("Metrics") }
if (!require(aspace)) { install.packages("aspace") }

load.from.sql <- function(sql)
{
  conn <- dbConnect(MySQL(), user='root', password='root', dbname='scraper', host='localhost')
  on.exit(dbDisconnect(conn))
  rs <- dbSendQuery(conn, sql)
  result <- fetch(rs, n=-1)
  dbClearResult(rs)
  
  return(result)
}

city_names <- c('London', 'Birm', 'Glas', 'Liv', 'Leeds', 'Manch')
city_lats <- c(51.506, 52.481, 55.865, 53.411, 53.796, 53.481)
city_longs <- c(-0.109, -1.9, -4.258, -2.978, -1.548, -2.237)
city_sq_mi <- c(671, 598.9, 368.5, 199.6, 364.5, 630.3)
cities <- data.frame(names=city_names, lats=city_lats, longs=city_longs, sq_mi=city_sq_mi)

cities$km_radius <- sqrt(cities$sq_mi / pi) / 0.621371
cities$lat_radius <- abs(cities$km_radius / 110.54)
cities$long_radius <- abs(cities$km_radius / (111.32 * cos_d(cities$lats)))
cities$deg_radius <- (cities$lat_radius + cities$long_radius) / 2

cutoff <- 0.8

# VAR NAMES
# x is the poorly-named dataset we load from the database
# df is used as the name of multiple output data frames

x <- load.from.sql("select id, monthly_price, is_share, num_bedrooms, latitude, longitude, let_agreed, furnishing, date_available, date_advertised, sent_pol, sent_subj, sent_pos, sent_neg, initial_timestamp, latest_timestamp, test_train, availability_lag from property where usable=1 and removed=0 and availability_lag <= 21;")

x$date_available <- as.Date(x$date_available)
x$date_advertised <- as.Date(x$date_advertised)
x$initial_timestamp <- as.Date(x$initial_timestamp)
x$latest_timestamp <- as.Date(x$latest_timestamp)

# the '-1' in the line below is because all 'latest_timestamp' values are from after midnight the day after
x$time_to_let_agreed <- as.integer(ifelse(x$let_agreed == 1, x$latest_timestamp - x$date_advertised - 1, NA))
x$furnishing <- (x$furnishing == "Furnished")

x$is_london <- (sqrt((x$latitude - cities$lats[1])^2 + (x$longitude - cities$longs[1])^2) < cities$deg_radius[1])
x$is_other_city <- FALSE
for (city in 2:nrow(cities))
{
  x$is_other_city <- x$is_other_city | (sqrt((x$latitude - cities$lats[city])^2 + (x$longitude - cities$longs[city])^2) < cities$deg_radius[city])
}

cat(paste(nrow(x), "properties\n"))
cat(paste(sum(x$let_agreed == 1)), "rented within two weeks\n")
cat(paste(sum(x$let_agreed == 0)), "did not rent within two weeks\n")

max_time_to_let_agreed = max(x$time_to_let_agreed, na.rm=TRUE)
x$time_to_let_agreed <- ifelse(is.na(x$time_to_let_agreed), max_time_to_let_agreed + 1, x$time_to_let_agreed)
#hist(x$predicted_time_to_let_agreed)

set.seed(12345)
range <- -4:4
df <- data.frame(sep=NA, group=NA, accuracy=NA)
for (sep in range)
{
  print(sep)

  train <- x[x$test_train < cutoff,]
  test <- x[x$test_train >= cutoff,]
  
  #lm_model <- lm(time_to_let_agreed ~ is_share + num_bedrooms + availability_lag + monthly_price + furnishing,
  #               data=train)
  
  rf_model <- randomForest(time_to_let_agreed ~ is_share + num_bedrooms + availability_lag + monthly_price + furnishing,
                 data=train, type="regression", nodesize=7)
  
  preds <- predict(rf_model, x)
  x$predicted_time_to_let_agreed <- preds
  x$better_value <- x$time_to_let_agreed < x$predicted_time_to_let_agreed + sep
  
  test <- x[x$test_train >= cutoff,]
  train <- x[x$test_train < cutoff,]
  
  # #Equalize the categories
  # train2 <- train[!train$better_value,]
  # train3 <- train[train$better_value,]
  # train2 <- head(train2, min(nrow(train2), nrow(train3)))
  # train3 <- head(train3, min(nrow(train2), nrow(train3)))
  # train <- rbind(train2, train3)

  print(c(sum(test$better_value), sum(!test$better_value), sum(train$better_value), sum(!train$better_value)))

  rf_model <- randomForest(factor(better_value) ~ is_share + num_bedrooms + availability_lag + monthly_price + furnishing + latitude + longitude,
                           data=train, type="classification", nodesize=7)
  preds <- predict(rf_model, test)
  df <- rbind(df, c(sep, 1, mean(preds==test$better_value)))
  
  rf_model <- randomForest(factor(better_value) ~ sent_pol + sent_subj + latitude + longitude,
                           data=train, type="classification", nodesize=7)
  preds <- predict(rf_model, test)
  df <- rbind(df, c(sep, 2, mean(preds==test$better_value)))
  
  rf_model <- randomForest(factor(better_value) ~ is_share + num_bedrooms + availability_lag + monthly_price + furnishing + latitude + longitude + sent_pol + sent_subj,
                           data=train, type="classification", nodesize=7)
  preds <- predict(rf_model, test)
  df <- rbind(df, c(sep, 3, mean(preds==test$better_value)))
}
df <- df[-1,]

df$group <- factor(df$group)
df$Features <- NA
df$Features[df$group==1] <- "Metadata"
df$Features[df$group==2] <- "Sentiment Analysis"
df$Features[df$group==3] <- "Metadata + Sentiment Analysis"

p <- ggplot(df, aes(x=sep,y=accuracy))
p <- p + geom_line(aes(color=Features), size=2)
p <- p + xlab("Cutoff relative to prediction (-4 to +4 days)")
p <- p + ylab("Prediction accuracy")
p <- p + ggtitle("Prediction Accuracy by Input Features")
p
ggsave("accuracy comparisons.png", width=8, height=6, units="in", dpi=200)

df2 <- data.frame(offset=NA, diff=NA)
for (sep in range)
{
  df2 <- rbind(df2, c(sep, df$accuracy[df$group==3 & df$sep==sep] - df$accuracy[df$group==1 & df$sep==sep]))
}
df2 <- df2[-1,]
df2
write.csv(df2, "improvement.csv")
cat(mean(df2$diff))

p <- ggplot(df2, aes(x=offset,y=diff))
p <- p + geom_line(size=2, color="blue")
p <- p + xlab("Cutoff relative to prediction (-4 to +4 days)")
p <- p + ylab("Improvement in Prediction accuracy")
p <- p + ggtitle("Improvement in Prediction Accuracy from Incorporating Sentiment Analysis")
p
ggsave("improvement.png", width=8, height=6, units="in", dpi=200)
