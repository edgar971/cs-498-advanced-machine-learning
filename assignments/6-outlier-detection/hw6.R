setwd('~/dev/school/cs-498-advanced-machine-learning/assignments/6-outlier-detection/data')
housing_data = read.table("housing.txt", header=FALSE, col.names = c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"))

######################## Question 1 ########################

# Create the model
orig_model = lm(MEDV ~ ., data = housing_data)

summary(orig_model)
plot(rstandard(orig_model))

## Search for outliers manually
c_distance = cooks.distance(orig_model)
c_distance = data.frame(1:nrow(housing_data), c_distance)
colnames(c_distance) = c("idx", "value")

par(mfrow=c(2,2))
plot(orig_model)
par(mfrow=c(1,1))

plot(c_distance$idx ~ c_distance$value)
text(c_distance$idx ~ c_distance$value, labels=idx, data=c_distance, cex=.75, font=2)

######################## Question 2 ########################
# Remove outliers
cleaned_data = housing_data[-c(365,366,369,373,370,413),]
row.names(cleaned_data) <- 1:nrow(cleaned_data)
new_model = lm(MEDV ~ ., data= cleaned_data)
summary(new_model)

par(mfrow=c(2,2))
plot(new_model)
par(mfrow=c(1,1))

# Cooks distance with outliers removed
c_distance = cooks.distance(new_model)
c_distance = data.frame(1:nrow(cleaned_data), c_distance)
colnames(c_distance) = c("idx", "value")

plot(c_distance$idx ~ c_distance$value)
text(c_distance$idx ~ c_distance$value, labels=idx, data=c_distance, cex=.75, font=2)

######################## Question 3 ########################
bc = boxcox(MEDV ~ .,  data=cleaned_data)
lambda = bc$x[which.max(bc$y)]

cat("Lambda is", lambda)

######################## Question 4 ########################
q4_data = cleaned_data
q4_data$MEDV = q4_data$MEDV ** lambda
new_fit = lm(MEDV ~ ., data=q4_data)
plot(rstandard(new_fit))
predicted = fitted(new_fit) ** (1/lambda)

plot(predicted ~ cleaned_data$MEDV)


