##predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
##Submissions are evaluated on Root-Mean-Squared-Error (RMSE)

#set WD
setwd("/Users/User/Desktop/PredictHousePrices/Data")

#load libraries
library(ggplot2)
library(naniar)
library(dplyr)
library(gridExtra)

#load Data
train = read.csv("train.csv", stringsAsFactors = FALSE) #using strings as factors=F cos train and test datasets have different factor numbers for some features
test = read.csv("test.csv", stringsAsFactors = FALSE)  #missing sale price

#removing saleprice from train dataset
SalePrice <- train$SalePrice
train <- select(train, -SalePrice)

#combining datasets
full <- rbind(train, test)

#fill NA with missing and convert chr columns to factors
for (col in colnames(full)){ #using for loop
  if (typeof(full[,col]) == "character"){ #if the type of column = character, then we replace NA with missing
    new_col = full[,col]  #obtain the character column
    new_col[is.na(new_col)] = "missing" #replace NA with missing
    full[col] = as.factor(new_col) #change the column to factor
  }
}

#separating train data from full
train <- full[1:nrow(train),] #getting back the rows for train
train <- cbind(train, SalePrice)#combining the sale price back to train
test <- full[(nrow(train)+1):(nrow(full)),]

#Visualizing missing data
gg_miss_var(train, show_pct = F) + labs("Missing")
gg_miss_var(test, show_pct = F) + labs("Missing")

#replacing NA values under integer type col with -1
train[is.na(train)] <- -1
test[is.na(test)] <- -1

#EDA - trying to see a relationship between property size/area with sale price
p1 <- ggplot(train, aes(GrLivArea, SalePrice)) + geom_point(alpha = 0.5, color = 'blue') + theme_bw()
p2 <- ggplot(train, aes(LotArea, SalePrice)) + geom_point(alpha = 0.5,color = 'green') + theme_bw()
p3 <- ggplot(train, aes(LotFrontage, SalePrice)) + geom_point(alpha = 0.5,color = 'red') + theme_bw()
p4 <- ggplot(train, aes(GarageArea, SalePrice)) + geom_point(alpha = 0.5,color = 'gray') + theme_bw()
grid.arrange(p1, p2, p3, p4, ncol=2)

ggplot(train, aes(GrLivArea, SalePrice)) + geom_point(aes(color = Neighborhood), alpha=0.5) + 
  scale_x_continuous("GrLivArea") +
  scale_y_continuous("SalePrice") +
  theme_bw() + facet_wrap( ~ Neighborhood) + theme(legend.position="none")

#----------Identifying highly correlated predictors with SalePrice----------#
for (col in colnames(train)){ #start of loop function btw each column
  if(is.numeric(train[,col])){ #identify if the column is numeric, else ignore
    if( abs(cor(train[,col],train$SalePrice)) > 0.5){ #if the absolute number (i.e. numeric) of the correlation btw each col vs SalePrice > 0.5
      print(col) #print out the name of the column if the correlation no. is > 0.5
      print( cor(train[,col],train$SalePrice) ) #print out the correlation value
    }
  }
}

#----------Identifying lowly correlated predictors with SalePrice----------#
for (col in colnames(train)){ #start of loop function btw each column
  if(is.numeric(train[,col])){ #identify if the column is numeric, else ignore
    if( abs(cor(train[,col],train$SalePrice)) < 0.1){ #if the absolute number (i.e. numeric) of the correlation btw each col vs SalePrice > 0.5
      print(col) #print out the name of the column if the correlation no. is < 0.1
      print( cor(train[,col],train$SalePrice) ) #print out the correlation value
    }
  }
}

#----------Identifying if the features are correlated to each other-------#
cors = cor(train[ , sapply(train, is.numeric)]) #shows a correlation matrix
high_cor = which(abs(cors) > 0.6 & (abs(cors) < 1)) #identify the positions of high correlation features, counting from col down!
rows = rownames(cors)[((high_cor-1) %/% 38)+1] # %/% = integer division i.e. 5 %/% 2 = 2/ 38 BECAUSE it has 38 col & rows!
cols = colnames(cors)[ifelse(high_cor %% 38 == 0, 38, high_cor %% 38)] #modulus (x mod y) 5%%2 = 1 
vals = cors[high_cor]

cor_data = data.frame(cols=cols, rows=rows, correlation=vals)
cor_data #as none of them are >0.9, we dont have to remove any features

library(corrplot) 
corrplot(cors) #visualize correlation

#------Identify outliers for numeric cols-----#
for (col in colnames(train)){
  if(is.numeric(train[,col])){
    plot(density(train[,col]), main=col)
  }
} #data shows that spring and summer sells more houses than winter, and that salepx is right skewed, meaning some houses sell higher than avg

#-----Data transformation by combining variables----#


# Add variable that combines above grade living area with basement sq footage
train$total_sq_footage = train$GrLivArea + train$TotalBsmtSF
test$total_sq_footage = test$GrLivArea + test$TotalBsmtSF

# Add variable that combines above ground and basement full and half baths
train$total_baths = train$BsmtFullBath + train$FullBath + (0.5 * (train$BsmtHalfBath + train$HalfBath))
test$total_baths = test$BsmtFullBath + test$FullBath + (0.5 * (test$BsmtHalfBath + test$HalfBath))

#removing IDs
train <- select(train, -Id)
test <- select(test, -Id)

#######################################
#----------Predictive Model-----------#
#######################################
library(caret)
library(Metrics)
library(xgboost)

# Create custom summary function in proper format for caret
custom_summary <- function(data, lev = NULL, model = NULL){
  out = rmsle(data[, "obs"], data[, "pred"])
  names(out) = c("rmsle")
  out
}

# Create control object
control <- trainControl(method = "cv",  # Use cross validation
                       number = 5,     # 5-folds
                       summaryFunction = custom_summary                      
)


# Create grid of tuning parameters
grid <- expand.grid(nrounds=c(100, 200, 400, 800), # Test 4 values for boosting rounds
                    max_depth= c(4, 6),           # Test 3 values for tree depth
                    eta=c(0.1, 0.05, 0.025),      # Test 3 values for learning rate: 0.1, 0.05, 0.025
                    gamma= c(0.1),                #https://xgboost.readthedocs.io/en/latest/parameter.html for explanation
                    colsample_bytree = c(1), 
                    min_child_weight = c(1),
                    subsample = c(1))

#training and development of model

xgb_tree_model <- train(SalePrice~.,      # Predict SalePrice using all features
                        data=train,
                        method="xgbTree",
                        trControl=control, #for cross validation w control
                        tuneGrid=grid, 
                        metric="rmsle",     # Use custom performance metric
                        maximize = FALSE)   # Minimize the metric

#Analysis of results
xgb_tree_model$bestTune #tells us the best model, is a tree with depth 4, trained 400 rounds w learning rate 0.1 (eta)
xgb_tree_model$results #find the RMSLE from the above model here: RMSLE = 0.1327114; the lower the better!


varImp(xgb_tree_model) #identify which predictors are most impt to the model

#testing of dataset

test_predictions <- predict(xgb_tree_model, newdata=test)

submission <- read.csv("sample_submission.csv")
submission$SalePrice <- test_predictions
write.csv(submission, "home_prices_xgb_sub1.csv", row.names=FALSE)
