#KNN
### Clear environment
rm(list = ls())
#Q1
#Loading the built in dataset Iris
data(iris)
head(iris)

#check the datatype
str(iris)

#printing summary
summary(iris)

#check null values
colSums(is.na(iris))
#no missing value so need for ommission or imputation

#distribution plots
#load imp libraries
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(caret)
library(class) 

# Step 1: Remove the 1 categorical columns for creating distribution plots
df_numeric <- iris %>%
  select(where(is.numeric))

# Step 2: Plot boxplots for each numeric column
# Set up plotting layout: 4 rows × 4 columns per page
par(mfcol = c(2, 2))
for (i in 1:ncol(df_numeric)) {
  boxplot(df_numeric[[i]],
          main = paste("Boxplot of", names(df_numeric)[i]),
          col = "tomato", border = "black")
}

# Step 3: Plot histograms for each numeric column
for (i in 1:ncol(df_numeric)) {
  hist(df_numeric[[i]],
       main = paste("Histogram of", names(df_numeric)[i]),
       col = "steelblue", border = "white", breaks = 30)
}

#corr plot
# Compute correlation matrix
cor_matrix <- cor(na.omit(df_numeric))

# Plot using ggcorrplot
ggcorrplot(cor_matrix,
           hc.order = TRUE,         # hierarchical clustering
           type = "full",          # show lower triangle
           lab = TRUE,              # show correlation values
           lab_size = 3,
           colors = c("red", "white", "green"),
           title = "Correlation Matrix",
           ggtheme = theme_minimal())

##examples to see correlation b/w 2 variables
# Petal width and sepal length plot
ggplot(iris, aes(x = Petal.Width, y = Sepal.Length)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  theme_minimal()

# Petal width and petal length plot
ggplot(iris, aes(x = Petal.Width, y = Petal.Length)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "blue") +
  theme_minimal()

# Petal length and sepal width plot
ggplot(iris, aes(x = Petal.Length, y = Sepal.Width)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "blue") +
  theme_minimal()

#Q2
# use set.seed() to get the same partitions when re-running the R code.
set.seed(16)

## partitioning into training (70%) and validation (30%)
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData  <- iris[-trainIndex, ]

#Preprocess (center and scale)
pp <- preProcess(trainData[, 1:4], method = c("center", "scale"))

#apply pre-processing to both sets
trainData[,1:4] <- predict(pp,trainData[,1:4])
testData[,1:4] <- predict(pp,testData[,1:4])

pp <- preProcess(trainData[, 1:4], method = c("center", "scale"))
trainData[, 1:4] <- predict(pp, trainData[, 1:4])
testData[, 1:4]  <- predict(pp, testData[, 1:4])

# Define a grid of k values to test
knnGrid <- expand.grid(k = c(1, 3, 5, 7, 9, 11, 13))

#Q3
# Train KNN model
set.seed(16)
knn_model <- train(Species ~ ., 
                   data = trainData , 
                   method = "knn",
                   tuneGrid = knnGrid,
                   trControl = trainControl(method = "cv", number = 5))

# Show results
print(knn_model)
plot(knn_model)  # Plot Accuracy vs k


##after this we will run predictions on test dataset 
pred <- predict(knn_model, newdata = testData)

#Q4
# Confusion matrix
confusionMatrix(pred, testData$Species)

#Q5
# Two numeric features for plotting (original)
plotIrisdata <- iris %>% select(Sepal.Length, Sepal.Width, Species)

ggplot(plotIrisdata, aes(x = Sepal.Length, y = Sepal.Width, color = Species, shape = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Iris Species Scatterplot",
       x = "Sepal Length",
       y = "Sepal Width") +
  theme_minimal()

#Q6
# Step 1: Partition the data
set.seed(16)
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData  <- iris[-trainIndex, ]

# Step 2: Preprocess Sepal features (center and scale)
pp <- preProcess(trainData[, c("Sepal.Length", "Sepal.Width")], method = c("center", "scale"))
trainData_scaled <- trainData
trainData_scaled[, c("Sepal.Length", "Sepal.Width")] <- predict(pp, trainData[, c("Sepal.Length", "Sepal.Width")])

# Step 3: Prepare scaled plot data for scatterplot (Q5)
plotData <- iris[, c("Sepal.Length", "Sepal.Width")]
plotData_scaled <- predict(pp, plotData)
plotData_scaled <- data.frame(plotData_scaled, Species = iris$Species)


# Step 4: Create grid for decision boundary (Q6)
x_min <- min(trainData_scaled$Sepal.Length) - 0.5
x_max <- max(trainData_scaled$Sepal.Length) + 0.5
y_min <- min(trainData_scaled$Sepal.Width) - 0.5
y_max <- max(trainData_scaled$Sepal.Width) + 0.5

grid <- expand.grid(
  Sepal.Length = seq(x_min, x_max, length.out = 100),
  Sepal.Width = seq(y_min, y_max, length.out = 100)
)

# Step 5: Predict species on grid using KNN for k = 2 and k = 13
plot_knn_boundary <- function(k_val) {
  grid$Species <- knn(train = trainData_scaled[, c("Sepal.Length", "Sepal.Width")],
                      test = grid,
                      cl = trainData_scaled$Species,
                      k = k_val)
  
  ggplot() +
    geom_tile(data = grid, aes(x = Sepal.Length, y = Sepal.Width, fill = Species), alpha = 0.3) +
    geom_point(data = plotData_scaled, aes(x = Sepal.Length, y = Sepal.Width, color = Species, shape = Species), size = 3) +
    labs(title = paste("KNN Decision Boundary (k =", k_val, ")"),
         x = "Sepal Length",
         y = "Sepal Width") +
    theme_minimal()
}

# Step 6: Plot decision boundaries (Q6)
print(plot_knn_boundary(2))
print(plot_knn_boundary(13))

