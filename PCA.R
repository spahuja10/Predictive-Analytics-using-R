##PCA

##Importing wine data
setwd("C:/Predictive Analytics")

#load the data
wine <- read.csv("wine.csv", header = TRUE)

#summary
summary(wine)

#check for null values
colSums(is.na(wine))


# to see the distribution and outliers
#  Use R to plot a histogram for each of the quantitative variables. 
par(mfcol=c(4,4))
for (i in c(2:14))
  hist(wine[,i], main=names(wine)[i])

#boxplot to see % of Alcohol in each Wine Type
par(mfcol=c(1,1))
boxplot(wine$Alcohol ~ wine$Type, main = "Distribution of Alcohol Perc",
        xlab = "Type", ylab = "PercOfAlcohol")

#Boxplot for each variable except the categorical variable
par(mfcol=c(4,4))
for (i in c(2:14))
  boxplot(wine[,i], main=names(wine)[i])

# correlation analysis
options(digits = 1)
options(scipen=999)
cor(na.omit(wine[,-c(1)]))
plot(na.omit(wine[,-c(1)]))


# correlation heatmap
heatmap(cor(na.omit(wine[,-c(1)])), Rowv = NA, Colv = NA)


# heatmap with values
library(gplots)
heatmap(cor(na.omit(wine[,-c(1)])), Rowv = FALSE, Colv = FALSE, 
        dendrogram = "none", 
        cellnote = round(cor(na.omit(wine[,-c(1)])),2), 
        notecol = 'black', key = FALSE, margins = c(10,10))

# Create a principal component analysis without normalization
pcs <- prcomp(na.omit(wine[,-c(1)])) 
summary(pcs)

#how much a variable contribute to each PCA -- without normalization
round(pcs$rotation,5)


# scaling the data before PCA-- with normalization
pcs.cor <- prcomp(na.omit(wine[,-c(1)]), scale. = T)
summary(pcs.cor)

#how much a variable contribute to each PCA -- with normalization
round(pcs.cor$rotation,3)
