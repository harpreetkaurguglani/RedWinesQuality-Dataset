# RedWinesQuality-Dataset

Regression Analysis attempting to predict the quality of wine according to the several predictors included in the data set. 

## Exploratory data analysis

The Red Wine Dataset had 1599 rows and 12 columns. Here our response variable is 'quality' which is numeric variable as we need to perform regression analysis, and the rest of the predictors and variables are numerical variables which reflect the physical and chemical properties of the wine.There are no NAs in the dataset.The data is splited in 7:3 ratio for training and testing purpose split.The summary looks good.

The Quality is rated from 1-10 ratings and the ratings with 6.5 and above is considered as a good quality. Data is not having equal quality amongst the different category/the quality. But since we are performing the regression analysis we would move froward with the exact dataset without categorizing it. 
## Analysis objective
Which model is most likely to provide the lowest MSE in the long-run? Along with this the analysis would also direct towards which variables are having a powerful impact on identifying the quality of the red wine.


## Summary

If we consider the estimation of MSE in each case, the smallest occurs when using random forests (providing an expected long-run MSE of around 0.33 via the test set, and similar OOB estimate on the training). In fact, it stands out as substantially better in comparison to the other models, which all have test MSEs of between 0.39-0.44.

If consulting with a company on this data set, and it was clear that they had interest in a simplified model, we might give consideration to the LASSO model with the “within one standard error” lambda value, as the number of predictors is highly reduced even though the predictive power sits about equivalent to the full linear model. But, if instead prediction was of primary importance, then the random forests model would be clearly the route to go.
