getwd()
#setwd("C:/Users/soukaev/Desktop/Admin/Professional Development/University of Illinois/3 - Spring 2019/CS 498 (Applied Machine Learning)/Week 7/data")
setwd('~/dev/school/cs-498-advanced-machine-learning/assignments/6-outlier-detection/data')
housing_data = read.table("housing.txt", header=FALSE, col.names = c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"))

plot_fitted_resid = function(model, pointcol = "dodgerblue", linecol = "darkorange") {
  plot(fitted(model), rstandard(model), 
       col = pointcol, pch = 20, cex = 1.5,
       xlab = "Fitted", ylab = "Residuals")
  abline(h = 0, col = linecol, lwd = 2)
}


######################## Explore ########################


######################## Question 1 ########################
orig_model = lm(MEDV ~ ., data = housing_data)

par(mfrow=c(2,2))
plot(orig_model)
par(mfrow=c(1,1))

plot_fitted_resid(orig_model)

possible_outliers = as.numeric(names(resid(orig_model)[cooks.distance(orig_model) > 4 / length(cooks.distance(orig_model))]))

######################## Question 2 ########################
influential = as.numeric(names(resid(orig_model)[cooks.distance(orig_model) > 4 / length(cooks.distance(orig_model))]))
cat("Removing", length(influential), " outliers")
cleaned_housing_data = housing_data[-influential, ]
new_model = lm(MEDV ~ ., data = cleaned_housing_data)

par(mfrow=c(2,2))
plot(new_model)
par(mfrow=c(1,1))

plot_fitted_resid(new_model)

######################## Question 3 ########################

require(MASS)
bc = boxcox(new_model)
lambda = bc$x[which.max(bc$y)]
lambda

######################## Question 4 ########################

transformed_model = lm(((MEDV ^ lambda - 1)/lambda) ~ ., data = cleaned_housing_data)

par(mfrow=c(2,2))
plot(transformed_model)
par(mfrow=c(1,1))

plot_fitted_resid(transformed_model)

plot(((transformed_model$fitted.values)*lambda)^(1/lambda), cleaned_housing_data$MEDV)

# Option 2
predicted <- fitted(transformed_model) ** (1/lambda)
plot(predicted ~ cleaned_housing_data$MEDV)

summary(new_model)

