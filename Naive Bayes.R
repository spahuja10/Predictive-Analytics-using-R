### Clear environment
rm(list = ls())


            ######NAIVE BAYES CLASSIFICATION MODELING#####


##load important libraries
library(ggplot2)
library(caret)
library(e1071)
library(corrplot)
library(dplyr)

#importing data
setwd("C:/Predictive Analytics")

#data
accidents <- read.csv("accidentsFull.csv", header = TRUE)
print(accidents)

#create injury(1 or 2)/non-injury(0) binary variable
accidents$INJURY <- ifelse(accidents$MAX_SEV_IR %in% c(1,2),1,0)
print(accidents)

##Q1- Explore the data
summary(accidents)

#missing data??
colSums(is.na(accidents))##no missing data

##visualizations
#HISTOGRAM
numeric_vars <- names(accidents)[sapply(accidents, is.numeric)]
#plotting area
par(mfrow = c(5, 5))  # 5x5 grid for 25 plots

for (var in numeric_vars) {
  if (is.numeric(accidents[[var]])) {
    hist(accidents[[var]],
         main = paste("Histogram of", var),
         xlab = var,
         col = "steelblue")
  }
}


#BOXPLOT
par(mfrow = c(5, 5))
for (var in numeric_vars[1:25]) {
  boxplot(accidents[[var]],
          main = paste("Boxplot of", var),
          col = "orange",
          border = "darkblue")
}

##HEATMAP
par(mfrow = c(1, 1))
cor_matrix <- cor(accidents[, numeric_vars], use = "complete.obs")

# Plot correlation heatmap
corrplot(cor_matrix,
         method = "color",       
         type = "full",         
         tl.cex = 0.8,           
         tl.col = "black",       
         col = colorRampPalette(c("darkred", "white", "darkblue"))(200),
         title = "Correlation Heatmap of Numeric Variables",
         mar = c(0, 0, 1, 0))


#Q2
##Create a table & Calculate Probability
table(accidents$INJURY)
prop.table(table(accidents$INJURY))
"So if no predictors or no further information is available
then we can make our predictions based on how often Injury happened by
seeing INJURY which is our response i.e., determined by checking if the Probability of INJURY
= 1 is more or Probabilty of Injury = 0.
We can see that Probability of Injury = 1 or YES is more 
as compared to 0 (NO). This tells that the dataset shows that the injuries
occured in almost 51% of the cases which makes it a majority class.
When we have no context or predictors available to decide, the optimal strategy
is to decide the most frequent outcome/class."

#Q3
#a)
##Selecting 12 random records
randomData <- accidents [1:12, c("INJURY","WEATHER_R","TRAF_CON_R")]

#Creating pivot table using specified variables
ftable(randomData$INJURY, randomData$WEATHER_R, randomData$TRAF_CON_R)


#b)
#Output for Injury = Yes
"      0 1 2
1   1  2 0 0
    2  1 0 0"
##Computing Probabilities using Bayes Theorem for 6 combinations
#P(Injury = 1 | WEATHER_R = 1, TRAF_CON_R = 0)
p1 <- 2/(2+1)
p1
#P(Injury = 1 | WEATHER_R = 1, TRAF_CON_R = 1)
p2 <- 0 / (0+1)
p2
#P(Injury = 1 | WEATHER_R = 1, TRAF_CON_R = 2)
p3 <- 0 / (0+1)
p3
#P(Injury = 1 | WEATHER_R = 2, TRAF_CON_R = 0)
p4 <- 1 / (1+5)
p4
#P(Injury = 1 | WEATHER_R = 2, TRAF_CON_R = 1)
p5 <- 0 / (0+1)
p5
#P(Injury = 1 | WEATHER_R = 2, TRAF_CON_R = 2)
p6 <- 0 / (0+0)
p6

#c
randomData <- randomData %>%
  mutate(prob = case_when(
    WEATHER_R == 1 & TRAF_CON_R == 0 ~ p1,
    WEATHER_R == 1 & TRAF_CON_R == 1 ~ p2,
    WEATHER_R == 1 & TRAF_CON_R == 2 ~ p3,
    WEATHER_R == 2 & TRAF_CON_R == 0 ~ p4,
    WEATHER_R == 2 & TRAF_CON_R == 1 ~ p5,
    WEATHER_R == 2 & TRAF_CON_R == 2 ~ p6
  ),
  prediction = ifelse(prob >= 0.5, "Yes", "No"))

randomData[, c("WEATHER_R", "TRAF_CON_R", "prob", "prediction")]
"1          1          0 0.6666667        Yes
2          2          0 0.1666667         No
3          2          1 0.0000000         No
4          1          1 0.0000000         No
5          1          0 0.6666667        Yes
6          2          0 0.1666667         No
7          2          0 0.1666667         No
8          1          0 0.6666667        Yes
9          2          0 0.1666667         No
10         2          0 0.1666667         No
11         2          0 0.1666667         No
12         1          2 0.0000000         No"


#d
#P(INJURY =1 | WEATHER_R = 1,TRAF_CON_R =1)
#Calculating manually (Changes: Just replace I= 1 and 0 with "Yes" or "NO")
#P(1|W=1,T=1)= (P(1)*P(W=1|1)*P(T=1|1))/(P(1)*P(W=1|1)*P(T=1|1))+ (P(0)*P(W=1|0)*P(T=1|0))
#P(1)
pa = 3/12
#P(W=1|1)
pb=2/3
#P(T=1|1)
pc=0/3
#P(0)
pd=9/12
#P(W=1|0)
pe=3/9
#P(T=1|0)
pf=2/9
##Bayes theorem
Probability= (pa*(pb*pc)/((pa*(pb*pc))+ (pd*(pe*pf))))
Probability  ##0

##e
randomData$INJURY <- as.factor(randomData$INJURY)
randomData$WEATHER_R <- as.factor(randomData$WEATHER_R)
randomData$TRAF_CON_R <- as.factor(randomData$TRAF_CON_R)

nb_randomData <- naiveBayes(INJURY ~ WEATHER_R +TRAF_CON_R, data = randomData)

# Predict class labels
pred.class <- predict(nb_randomData, newdata = randomData, type = "class")

# Predict probabilities
pred.prob <- predict(nb_randomData, newdata = randomData, type = "raw")

# Combine original data with predictions
results <- cbind(randomData, prob_0 = pred.prob[, "0"], prob_1 = pred.prob[, "1"], Predicted = pred.class)

# View the results
print(results)
"No the above predictions are different as comapred to Bayes
classification, it classified three accidents which caused Injury
while the Naive Bayes Classifier gave that no accident caused any injury
as the Injury = 1 in Naive Bayes Classifies has probability less than
our cut-off of 0.5, so no accident lead to injuries.
"

#Q4

#Sample row numbers randomly.
ntrain.index <- sort(sample(ntotal, ntrain))  
train.df <- bank.df[ntrain.index, ]
valid.df <- bank.df[-ntrain.index, ]

#a)" We will include below variables
"- SPD_LIM: Speed Limit, miles per hour
- WRK_ZONE: 1=yes, 0=no
- WEATHER_R: 1=no adverse conditions, 2=rain,snow or other adverse condition
- TRAF_CON_R: Traffic control device: 	0=none, 1=signal, 2=other (sign, officer …)
- SUR_CON
- TRAF_WAY
- ALIGN_I
- PROFIL_I_R
- LGTCON_I_R
- INT_HWY
- REL_JCT_I_R
- REL_RWY_R
- WKDY_I_R
- HOUR_I_R

"
#b)
predictors <- c("SPD_LIM", "WEATHER_R", "TRAF_CON_R", "SUR_COND", "TRAF_WAY",
                "ALIGN_I", "PROFIL_I_R", "LGTCON_I_R", "WRK_ZONE", "INT_HWY",
                "RELJCT_I_R", "REL_RWY_R", "WKDY_I_R", "HOUR_I_R")

#all variables should be categorical so we need to use as.factor
train.df[predictors] <-lapply(train.df[predictors], as.factor)
valid.df[predictors] <-lapply(valid.df[predictors], as.factor)
train.df$INJURY <- as.factor(train.df$INJURY)
valid.df$INJURY <- as.factor(valid.df$INJURY)
nrow(valid.df)
##train the model using Naive Bayes classifier
nb_full <- naiveBayes(INJURY ~ ., data = train.df[, c("INJURY", predictors)])

#predict
pred.valid <-predict(nb_full, valid.df)

#Confusion Matrix
table(Predicted = pred.valid, Actual = valid.df$INJURY)

#c
##error rate
error_rate <- mean(pred.valid != valid.df$INJURY)
error_rate

#d % improvement over naive bayes
majority_class <- names(which.max(table(train.df$INJURY)))
naive_pred <- rep(majority_class, nrow(valid.df))

naive_error <- mean(naive_pred != valid.df$INJURY)

percent_improvement <- (naive_error - error_rate) / naive_error * 100
percent_improvement

"as the naive Bayes error is low as compared to Naive rule error 
(used as a benchmark), we can say that our model performs better
as comapred to naive rule error.Percent Inprovement is approx 6%."

#e
"Training dataset has no records of Injury = No and SPD_LIM = 5,
when the model is asked to give the probabilities in this case it will
give the output as 0. And as in naive Bayes we multiple the 
conditional probabilities, any zero makes the complete value as 0.
This is calles as Zero-Frequency Problem in Naive Bayes. 
Due to this reason P(INJURY = No | SPD_LIM =5) = 0.
From our research we found that this issue can be solved using the Laplace 
Smoothing were we can prevent the zero probabilities by adding a small constant 
to every category, this will make sure that every category will have atleast a 
tuny probabilities and this makes model more robust."
