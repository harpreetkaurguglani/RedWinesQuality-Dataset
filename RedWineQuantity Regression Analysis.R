---
title: 'RedWinesQuality-Regression Analysis'
author: "Harpreet Kaur"
date: "DEC 2021"
output:
  pdf_document: default
  html_document: default
---

```{r , results='hide'}
data= read.csv("winequality-red.csv")
attach(data)
names(data)
summary(data)
sum(is.na(data))
str(data)
```


```{r, results='hide'}
plot(data)
```

```{r}
summary(data$quality)
table(data$quality)
```

It seems that the full variables linear regression is not suitable for this data set. The plot from above indicates sufficiently high multi-collinearity.

Performing a correlation test would indicate that the following variables have a higher correlation to Wine Quality.

1. Alcohol
2. Sulphates(log10)
3. Volatile Acidity
4. Citric Acid


### LR model

```{r}

model1=lm(quality~.,data = data)
summary(model1)
plot(model1)
```

In terms of diagnostics, since our response variable is not, in fact, continuous, there is little chance that a linear model is “appropriate” in any strict sense. We can confirm this in the residuals vs fitted plot. So, we should be very careful about the interpretations surrounding the coefficient estimators in the following summary but note that violation of some of the assumptions of inferential linear regression does not negate the potential utility of the model for prediction, or even interpretation of aspects of the model. It negates the meaningfulness of, for example, the p-values from the summary.

#### Applying Backward Variable Selection 
From all the above variable lets remove residual.sugar

```{r}
sampleSize <- round(nrow(data)*0.7)
n <- sample(seq_len(sampleSize), size = sampleSize)
train_data <- data[n,]
test_data <- data[-n,]
```

```{r}
model_red1 <- lm(quality ~ fixed.acidity + volatile.acidity + citric.acid + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol,data = train_data)
summary(model_red1)

```

```{r}
#MSE of the test LR model
lm.pred = predict(model_red1, test_data)

mse_lm=mean((test_data$quality-lm.pred)^2)
```


It can be observed there are many predictor variables exhibits insignificance.The adjusted R-squared also performs poor result. 

###### Linear Regression Assumption Check

i) Errors are Normally distributed

```{r}
hist(model_red1$residuals)
plot(model_red1)
```

These are kind of normally distributed with outliers.

ii) Errors have zero mean and constant Variance

```{r}
mean(model_red1$residuals)
var(model_red1$residuals)
```

iii) Multi-collinearity exist as stated above as well and can be observed from the plot of the data.


```{r}
model.back_red <- lm(formula = quality ~ volatile.acidity + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + pH + sulphates + alcohol, data = train_data)
summary(model.back_red)
```

Removing free.sulfur.dioxide as well as it is not significant as per the p-value which can observed from the above output.

```{r}
model.back_red <- lm(formula = quality ~ volatile.acidity + chlorides + total.sulfur.dioxide + pH + sulphates + alcohol, data = train_data)
summary(model.back_red)

```

The Multiple R^2 is not significant enough to consider the model.
It is not recommended to model the data using linear regression.

```{r}
#MSE of the test LR model
lm_back.pred = predict(model.back_red, test_data)

mse_back=mean((test_data$quality-lm_back.pred)^2)
```

### randomForest

```{r}
library(randomForest)
```

```{r}
set.seed(123)
rf_model=randomForest(quality~.,data=train_data)
rf_model
```

```{r}
# test MSE
rf_pred=predict(rf_model,test_data)
mse_rf=mean((test_data$quality-rf_pred)^2)
```

```{r}
varImpPlot(rf_model)
```

In random forest models, alcohol is the most important variable in deciding the quality of the wine.

The above plot indicates that alcohol, sulphates, volatile, acidity, total.sulfur.dioxide and density are important predictors.

Prediction of high-quality wines with Random Forest model is good with less problem of overfitting and with high accuracy; because of the aggregation of decision trees. To pick a desired quality wine, it is always important to check few parameters mentioned above from the varImpPlot.

### Bagging

```{r}
# Tuning mtry value
bag_fit=randomForest(quality~.,data=train_data, ntree=500, mtry=6, importance=TRUE, proximity=TRUE)
print(bag_fit)
plot(bag_fit)
```

```{r}
# test MSE for bagging model
bag_pred=predict(bag_fit,test_data)
mse_bag=mean((test_data$quality-bag_pred)^2)
```

### Lasso

```{r}
library(glmnet)
set.seed(123)
data= read.csv("winequality-red.csv")
train_ind = sample.int(n=nrow(data), size=floor(0.7*nrow(data)), replace=F)
train = data[train_ind,]
test = data[-train_ind,]
# prepare the data for modeling
x.train = model.matrix(quality ~. , train)[,-1]
y.train = train$quality
x.test = model.matrix(quality ~. , test)[,-1]
y.test = test$quality
```


```{r}

cv.lasso = cv.glmnet(x.train, y.train, alpha=1)

par(mfrow=c(1,1))
bestlam =cv.lasso$lambda.min
lam1se =cv.lasso$lambda.1se
bestlam
lam1se
plot(cv.lasso)
```

we can consider the estimated best case (with 11 predictors) or a further reduced model (with 6 predictors).

```{r}
lasso.best = glmnet(x.train, y.train, alpha=1, lambda=bestlam)
coef(lasso.best)
lasso.pred = predict(lasso.best, x.test)
# test MSE for lasso model
mse_lasso=mean((y.test-lasso.pred)^2)
```

#### Collected Information:

```{r}
# **These might change with final execution depending on the data split
MSEs=c(mse_rf,mse_bag,mse_lasso,mse_lm,mse_back)
model_name=c('Random Forest','Bagging','Lasso','LR model','Backward LR model')
MSE_df=data.frame(MSEs,model_name)
MSE_df
```


This shows that the Random Forest and Lasso has least MSE for the same dataset. We can conclude that these model performs better atleast in comparision to the LR model.Their is improvement in the MSE from every obtained model which shows that what kind of impact each of them is leading to. Lasso helped in been more confident about the results from the backward LR model. The variable important too has helped us in interpreting and comparing the results from the backward selection model.

If we consider the estimation of MSE in each case, the smallest occurs when using random forests (providing an expected long-run MSE of around 0.33 via the test set, and similar OOB estimate on the training). In fact, it stands out as substantially better in comparison to the other models, which all have test MSEs of between 0.39-0.44.

If consulting with a company on this data set, and it was clear that they had interest in a simplified model, we might give consideration to the LASSO model with the “within one standard error” lambda value, as the number of predictors is highly reduced even though the predictive power sits about equivalent to the full linear model. But, if instead prediction was of primary importance, then the random forests model would be clearly the route to go.
