getwd()
setwd("C:/Users/soukaev/Desktop/Admin/Professional Development/University of Illinois/3 - Spring 2019/CS 498 (Applied Machine Learning)/Week 7/data")
housing_data = read.table("housing.data", header=FALSE, col.names = c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"))

plot_fitted_resid = function(model, pointcol = "dodgerblue", linecol = "darkorange") {
  plot(fitted(model), rstandard(model), 
       col = pointcol, pch = 20, cex = 1.5,
       xlab = "Fitted", ylab = "Residuals")
  abline(h = 0, col = linecol, lwd = 2)
}

######################## Question 1 ########################
orig_model = lm(MEDV ~ ., data = housing_data)
# plot(orig_model$fitted.values, housing_data$MEDV)
# summary(orig_model)

par(mfrow=c(2,2))
plot(orig_model)
par(mfrow=c(1,1))

# Residuals vs Fitted values plot indicates non-linear relationship
# Normal Q-Q plot indicates some deviation from normal distribution
# Scale-Location shows different spread along the range of predictors
# Residuals vs Leverage shows

# Plot standardized residuals vs fitted values
plot_fitted_resid(orig_model)

# Remove residuals that are more than 4 standard deviations from the mean
# which(rstandard(orig_model) > 4)
# housing_data = housing_data[-which(rstandard(orig_model) > 4), ]

# List values with high leverage. We use twice the average leverage as the cutoff for high leverage
# hatvalues(orig_model)[hatvalues(orig_model) > 2 * mean(hatvalues(orig_model))]

# Check for any influential observations. Report any observations you determine to be influential.
# Here we consider observation to be influential if its cooks distance is higher than 4/observations
resid(orig_model)[cooks.distance(orig_model) > 4 / length(cooks.distance(orig_model))]

# Explanation: Cook's distance shows the influence of each observation on the fitted response values
# We can also see that residuals that are more than 3 standard deviations from the mean are included in the above values
which(rstandard(orig_model) > 3)


######################## Question 2 ########################
influential = as.numeric(names(resid(orig_model)[cooks.distance(orig_model) > 4 / length(cooks.distance(orig_model))]))
cleaned_housing_data = housing_data[-influential, ]
new_model = lm(MEDV ~ ., data = cleaned_housing_data)

par(mfrow=c(2,2))
plot(new_model)
par(mfrow=c(1,1))

plot_fitted_resid(new_model)

######################## Question 3 ########################

require(MASS)
boxcox(new_model)
bc = boxcox(new_model)
lambda = bc$x[which.max(bc$y)]
lambda

######################## Question 4 ########################

transformed_model = lm(((MEDV ^ lambda - 1)/lambda) ~ ., data = cleaned_housing_data)

# Check standardized residuals
par(mfrow=c(2,2))
plot(transformed_model)
par(mfrow=c(1,1))

plot_fitted_resid(transformed_model)

# Should clean up more or no?
# cleaned_housing_data = cleaned_housing_data[ -c(411), ]

# Need to transform fitted values back to compare with true house prices 
# Looks better now. Residuals vs Fitted values plot indicates linear relationship

plot(((transformed_model$fitted.values)*lambda)^(1/lambda), cleaned_housing_data$MEDV)

summary(new_model)
