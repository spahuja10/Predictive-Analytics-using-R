### Clear environment
rm(list = ls())

##install packages
#install.packages("neuralnet")

# Load required libraries
library(caret)
library(neuralnet)
library(forecast)
library(corrplot)

#load data
data("mtcars")
df <- mtcars
df

#summary statistics
summary(df)

# View structure of the dataset
str(df)

##check for missing values
colSums(is.na(df))

#Distribution Plots
numeric_vars <- c("mpg", "cyl", "disp", "hp", "drat", 
                  "wt", "qsec", "vs", "am", "gear", "carb")

# Set up 4x3 plotting layout
par(mfrow = c(4, 3))

# Generate histograms
for (var in numeric_vars) {
  hist(df[[var]],
       main = paste("Histogram of", var),
       xlab = var,
       col = "pink",
       border = "black")
}


# Correlation matrix
cor_matrix <- cor(df)
corrplot(cor_matrix, method="color", tl.cex=0.8)

##Confirm??
"as all the variables are numeric and binary vraibles are already coded as 0/1,
we aren't spltting the two binary variable am and vs into two columns as it might create
multicollinerity issues."

# partition
set.seed(28)
train.index <- sample(row.names(df), 0.7*dim(df)[1])  
valid.index <- setdiff(row.names(df), train.index)  
train.df <- df[train.index, ]
valid.df <- df[valid.index, ]

#pre-processing/normalise the data
norm.values <- preProcess(train.df, method="range")
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)


# Fit Model 1: Standard NN with 2 hidden layers
system.time({
  nn_model1 <- neuralnet(mpg ~ cyl + 
               disp + 
               hp + 
               drat + 
               wt + 
               qsec + 
               vs + 
               am + 
               gear + 
               carb,
            data = train.norm.df,
            hidden = c(5, 3),
            linear.output = TRUE
  )
})


plot(nn_model1)

# training data predictions(normalized)
training.prediction <- compute(nn_model1, train.norm.df)
# predictions on validation data
validation.prediction <- compute(nn_model1, valid.norm.df)

# RMSE of normalized training and validation data
Normalized_RMSE_train1=RMSE(training.prediction$net.result, train.norm.df$mpg)
Normalized_RMSE_train1
# [1]0.0502005
Normalized_RMSE_valid1=RMSE(validation.prediction$net.result, valid.norm.df$mpg)
Normalized_RMSE_valid1
# [1]0.1654885
# MAE of normalized training and validation data
Normalized_MAE_train1=MAE(training.prediction$net.result, train.norm.df$mpg)
Normalized_MAE_train1
# [1]0.03113468
Normalized_MAE_valid1=MAE(validation.prediction$net.result, valid.norm.df$mpg)
Normalized_MAE_valid1
# [1]0.1390743


# Re-scaling
min_price <- min(train.df$mpg)
max_price <- max(train.df$mpg)
                 
#Training data Rescaling and Evaluation
train.predictions.norm <- training.prediction$net.result
train.pred.original1 <- train.predictions.norm * (max_price - min_price) + min_price
train.actual1 <- train.df$mpg
train.MAE.rescaled1 <- mean(abs(train.pred.original1 - train.actual1))
train.MAE.rescaled1
##0.7316651
train.RMSE.rescaled1 <- sqrt(mean((train.pred.original1 - train.actual1)^2))
train.RMSE.rescaled1
##1.179712

##Validation data Resacling and Evalustion
valid.predictions.norm <- validation.prediction$net.result
valid.pred.original1 <- valid.predictions.norm * (max_price - min_price) + min_price
valid.actual1 <- valid.df$mpg
valid.MAE.rescaled1 <- mean(abs(valid.pred.original1 - valid.actual1))
valid.MAE.rescaled1
##3.268245
valid.RMSE.rescaled1 <- sqrt(mean((valid.pred.original1 - valid.actual1)^2))
valid.RMSE.rescaled1
##3.888981


# Displaying all calculated metrics for Model 1 (Optional but helpful)
cat("--- Model 1 Metrics ---")

cat("\nNormalized Training RMSE1:", Normalized_RMSE_train1)

cat("\nNormalized Validation RMSE1:", Normalized_RMSE_valid1)

cat("\nNormalized Training MAE1:", Normalized_MAE_train1)

cat("\nNormalized Validation MAE1:", Normalized_MAE_valid1)

cat("\nRescaled Training MAE1:", train.MAE.rescaled1)

cat("\nRescaled Training RMSE1:", train.RMSE.rescaled1)

cat("\nRescaled Validation MAE1:", valid.MAE.rescaled1)

cat("\nRescaled Validation RMSE1:", valid.RMSE.rescaled1, "\n")


# Fit Model 2: Standard NN with 4 hidden layers with 4 nodes
system.time({
  nn_model2 <- neuralnet(mpg ~ cyl + 
                           disp + 
                           hp + 
                           drat + 
                           wt + 
                           qsec + 
                           vs + 
                           am + 
                           gear + 
                           carb,
                         data = train.norm.df,
                         hidden = c(4, 4, 4, 4),
                         linear.output = TRUE
  )
})

plot(nn_model2)

# training data predictions(normalized)
training.prediction2 <- compute(nn_model2, train.norm.df)
# predictions on validation data
validation.prediction2 <- compute(nn_model2, valid.norm.df)

# RMSE of normalized training and validation data
Normalized_RMSE_train2=RMSE(training.prediction2$net.result, train.norm.df$mpg)
Normalized_RMSE_train2
# [1]0.04131931
Normalized_RMSE_valid2=RMSE(validation.prediction2$net.result, valid.norm.df$mpg)
Normalized_RMSE_valid2
# [1]0.1690154
# MAE of normalized training and validation data
Normalized_MAE_train2=MAE(training.prediction2$net.result, train.norm.df$mpg)
Normalized_MAE_train2
# [1]0.02963512
Normalized_MAE_valid2=MAE(validation.prediction2$net.result, valid.norm.df$mpg)
Normalized_MAE_valid2
# [1]0.1368105

#Training data Rescaling and Evaluation
train.predictions.norm2 <- training.prediction2$net.result
train.pred.original2 <- train.predictions.norm2 * (max_price - min_price) + min_price
train.actual2 <- train.df$mpg
train.MAE.rescaled2 <- mean(abs(train.pred.original2 - train.actual2))
train.MAE.rescaled2
##0.6863286
train.RMSE.rescaled2 <- sqrt(mean((train.pred.original2 - train.actual2)^2))
train.RMSE.rescaled2
##0.9710039

##Validation data Resacling and Evalustion
valid.predictions.norm2 <- validation.prediction2$net.result
valid.pred.original2 <- valid.predictions.norm2 * (max_price - min_price) + min_price
valid.actual2 <- valid.df$mpg
valid.MAE.rescaled2 <- mean(abs(valid.pred.original2 - valid.actual2))
valid.MAE.rescaled2
##3.215048
valid.RMSE.rescaled2 <- sqrt(mean((valid.pred.original2 - valid.actual2)^2))
valid.RMSE.rescaled2
##3.971862


# Displaying all calculated metrics for Model 1 (Optional but helpful)
cat("--- Model 2 Metrics ---")

cat("\nNormalized Training RMSE2:", Normalized_RMSE_train2)

cat("\nNormalized Validation RMSE2:", Normalized_RMSE_valid2)

cat("\nNormalized Training MAE2:", Normalized_MAE_train2)

cat("\nNormalized Validation MAE2:", Normalized_MAE_valid2)

cat("\nRescaled Training MAE2:", train.MAE.rescaled2)

cat("\nRescaled Training RMSE2:", train.RMSE.rescaled2)

cat("\nRescaled Validation MAE2:", valid.MAE.rescaled2)

cat("\nRescaled Validation RMSE2:", valid.RMSE.rescaled2, "\n")


# Fit Model 3: Standard NN with 3 hidden layers and 5 nodes
softplus <- function(x) {
  log(1 + exp(x))
}

system.time({
  nn_model3 <- neuralnet(mpg ~ cyl + 
                           disp + 
                           hp + 
                           drat + 
                           wt + 
                           qsec + 
                           vs + 
                           am + 
                           gear + 
                           carb,
                         data = train.norm.df,
                         hidden = c(5, 5, 5),
                         act.fct = softplus,
                         linear.output = TRUE
  )
})


plot(nn_model3)

# training data predictions(normalized)
training.prediction3 <- compute(nn_model3, train.norm.df)
# predictions on validation data
validation.prediction3 <- compute(nn_model3, valid.norm.df)

# RMSE of normalized training and validation data
Normalized_RMSE_train3=RMSE(training.prediction3$net.result, train.norm.df$mpg)
Normalized_RMSE_train3
# [1]0.03776578
Normalized_RMSE_valid3=RMSE(validation.prediction3$net.result, valid.norm.df$mpg)
Normalized_RMSE_valid3
# [1]0.2125682
# MAE of normalized training and validation data
Normalized_MAE_train3=MAE(training.prediction3$net.result, train.norm.df$mpg)
Normalized_MAE_train3
# [1]0.02963512
Normalized_MAE_valid3=MAE(validation.prediction3$net.result, valid.norm.df$mpg)
Normalized_MAE_valid3
# [1]0.1718694

#Training data Rescaling and Evaluation
train.predictions.norm3 <- training.prediction3$net.result
train.pred.original3 <- train.predictions.norm3 * (max_price - min_price) + min_price
train.actual3 <- train.df$mpg
train.MAE.rescaled3 <- mean(abs(train.pred.original3 - train.actual3))
train.MAE.rescaled3
##0.6964253
train.RMSE.rescaled3 <- sqrt(mean((train.pred.original3 - train.actual3)^2))
train.RMSE.rescaled3
##0.8874959

##Validation data Resacling and Evalustion
valid.predictions.norm3 <- validation.prediction3$net.result
valid.pred.original3 <- valid.predictions.norm3 * (max_price - min_price) + min_price
valid.actual3 <- valid.df$mpg
valid.MAE.rescaled3 <- mean(abs(valid.pred.original3 - valid.actual3))
valid.MAE.rescaled3
##4.038931
valid.RMSE.rescaled3 <- sqrt(mean((valid.pred.original3 - valid.actual3)^2))
valid.RMSE.rescaled3
##4.995352


# Displaying all calculated metrics for Model 1 (Optional but helpful)
cat("--- Model 3 Metrics ---")

cat("\nNormalized Training RMSE3:", Normalized_RMSE_train3)

cat("\nNormalized Validation RMSE3:", Normalized_RMSE_valid3)

cat("\nNormalized Training MAE3:", Normalized_MAE_train3)

cat("\nNormalized Validation MAE3:", Normalized_MAE_valid3)

cat("\nRescaled Training MAE3:", train.MAE.rescaled3)

cat("\nRescaled Training RMSE3:", train.RMSE.rescaled3)

cat("\nRescaled Validation MAE3:", valid.MAE.rescaled3)

cat("\nRescaled Validation RMSE3:", valid.RMSE.rescaled3, "\n")


# Final summary table of model performance (Using calculated variables)
model_comparison <- data.frame(
  Model = c("Model 1: 2 layers (5,3)",
            "Model 2: 4 layers (4x4)",
            "Model 3: 3 layers (5x5x5)"),
  
  Train_RMSE_Normalised = c(Normalized_RMSE_train1, Normalized_RMSE_train2, Normalized_RMSE_train3),
  Valid_RMSE_Normalised = c(Normalized_RMSE_valid1, Normalized_RMSE_valid2, Normalized_RMSE_valid3),
  
  Train_MAE_Normalised = c(Normalized_MAE_train1, Normalized_MAE_train2, Normalized_MAE_train3),
  Valid_MAE_Normalised = c(Normalized_MAE_valid1, Normalized_MAE_valid2, Normalized_MAE_valid3),
  
  Train_MAE_Rescaled  = c(train.MAE.rescaled1, train.MAE.rescaled2, train.MAE.rescaled3),
  Train_RMSE_Rescaled = c(train.RMSE.rescaled1, train.RMSE.rescaled2, train.RMSE.rescaled3),
  
  Valid_MAE_Rescaled  = c(valid.MAE.rescaled1, valid.MAE.rescaled2, valid.MAE.rescaled3),
  Valid_RMSE_Rescaled = c(valid.RMSE.rescaled1, valid.RMSE.rescaled2, valid.RMSE.rescaled3)
)

# Print the final table
print(model_comparison)
