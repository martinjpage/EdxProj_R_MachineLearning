### SUMMARY ###
# The below code loads and formats the Indian Liver Patient Dataset and builds 8 prediction models
# for classifying observations into either the liver disease (patient) or the non-liver disease groups. 
# The script should be read in conjunction with the report. Exploratory data analysis appears in the
# report. Only the model building is included in this script.

#=====#
### Load Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

### Read the data from the web (csv format)
url <- "https://raw.githubusercontent.com/martinjpage/HarvardCapstone/master/indian_liver_patient.csv"
liver <- read.csv(url)

### Format the Data
liver$Gender <- factor(liver$Gender) #gender is a categorical variable
liver$Dataset <- factor(liver$Dataset, levels = c(1,2), labels = c("Patient", "Control")) #dataset is a categorical variable; rename levels
liver <- na.omit(liver) #rows with NA values are removed

### Partition Validation Set (10%)
set.seed(1) #set seed for reproducibility 
inValid <- createDataPartition(y = liver$Dataset, times = 1, p = 0.1, list = FALSE)
validation <- liver[inValid,] #10% of the data are set aside for validation
modelling <- liver[-inValid,] #the remaining 90% of the data are used for modelling

### Partition Training (90%) and Testing Set (10%)
set.seed(2)
InTrain <- createDataPartition(y = modelling$Dataset, times = 1, p = 0.9, list = FALSE)
training <- modelling[InTrain,] #of the modelling data, 90% are used for training the models
testing <- modelling[-InTrain,] #of the modelling data, 10% are used for intermediate testing and tuning of the models

### Model Building
# 10 fold cross validation is used during model building
control <-  trainControl(method = "cv", number = 10, p = 0.9, allowParallel = TRUE)

### Model 1 - Logistic Regression
glmMod <-  train(Dataset ~ 0 + ., data = training, method = "glm", family = "binomial") #regression using all the features; remove intercept
p_glm <- predict(glmMod, newdata = testing) #predict using the model and the test set
glmAcc <- confusionMatrix(p_glm, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- tibble(Method = "Logistic Regression", Accuracy = round(glmAcc, 3)) #add accuracy to data frame

### Model 2 - GAM Model
set.seed(2) #set seed for reproducibility 
gamMod <- train(Dataset ~ ., data = training, method = "gamLoess", trControl = control, #train GamLoess model using all features
                tuneGrid = expand.grid(span = seq(0.15, 0.65, length = 10), degree = 1)) #try different spans; draw lines (=1)
p_gam <- predict(gamMod, newdata = testing) #test set prediction
gamAcc <- confusionMatrix(p_gam, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "GAM Model", Accuracy = round(gamAcc,3)))

### Model 3 - LDA Model
ldaMod <- train(Dataset ~ ., data = training, method = "lda", trControl = control) #train LDA model using all features
p_lda <- predict(ldaMod, newdata = testing) #test set prediction
ldaAcc <- confusionMatrix(p_lda, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "LDA Model", Accuracy = round(ldaAcc,3)))

### Model 4 - KNN
set.seed(4)
knnMod <- train(Dataset ~ ., data = training, method = "knn", trControl = control,  #train KNN model using all features
                tuneGrid = data.frame(k = seq(3,21,2))) #try different values of K
p_knn <- predict(knnMod, newdata = testing) #test set prediction
knnAcc <- confusionMatrix(p_knn, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "KNN Model", Accuracy = round(knnAcc,3)))

### Model 5 - Rpart model
rpartMod <- train(Dataset ~ ., data = training, method = "rpart", trControl = control, #train Rpart model using all features
                  tuneGrid = data.frame(cp = seq(0, 0.1, 0.01))) #try different values of the complexity parameter
p_rpart <- predict(rpartMod, newdata = testing) #test set prediction
rpartAcc <- confusionMatrix(p_rpart, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "Rpart Model", Accuracy = round(rpartAcc,3)))

### Model 6 - Random Forest
set.seed(6)
rfMod <- train(Dataset ~ ., data = training, method = "rf",     #train RF model using all features
               trControl = control, ntree = 100,                #build 100 random trees
               tuneGrid = data.frame(mtry = seq(1, 200, 25)))   #try different mtry values
p_rf <- predict(rfMod, newdata = testing) #test set prediction
rfAcc <- confusionMatrix(p_rf, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "Random Forest Model", Accuracy = round(rfAcc,3)))

### Model 7 - GBM Model
set.seed(8)
gbmMod <- train(Dataset ~ ., data = training, method = "gbm", verbose = FALSE, trControl = control, #turn off running print out to console
                tuneGrid = data.frame(expand.grid(n.trees = c(50, 100, 250), #try different number of trees
                                                  interaction.depth = seq(1,10, length.out = 3), #try different number of splits in each tree
                                                  shrinkage = c(.01, .1, .3), #try different levels of regularisation by shrinkage
                                                  n.minobsinnode = c(5, 10, 15)))) #try different number of observations in each leaf
p_gbm <- predict(gbmMod, newdata = testing) #test set prediction
gbmAcc <- confusionMatrix(p_gbm, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "GBM Model", Accuracy = round(gbmAcc,3)))

### Model 8 - Ensemble
p_all <- tibble(p_glm, p_gam, p_lda, p_knn, p_rpart, p_rf, p_gbm) #collect the predictions of all 7 individual models
ens_list <- apply(p_all,1,table) #count for each row the number of predictions for the patient and the control groups (the votes) 
ens_vote <- sapply(ens_list, which.max) #determine which group (liver disease or non-liver disease) has the most votes
p_ens <- factor(names(ens_vote)) #set final prediction for each row as a factor using the names of the groups
ensAcc <- confusionMatrix(p_ens, testing$Dataset)$overall["Accuracy"] #test set accuracy
acc_results <- bind_rows(acc_results, tibble(Method = "Ensemble Model", Accuracy = round(ensAcc,3)))

### Evaluate on Validation Set
#calculate the prediction on the final models with the validation set
pv_glm <- predict(glmMod, newdata = validation)
pv_gam <- predict(gamMod, newdata = validation)
pv_lda <- predict(ldaMod, newdata = validation)
pv_knn <- predict(knnMod, newdata = validation)
pv_rpart <- predict(rpartMod, newdata = validation)
pv_rf <- predict(rfMod, newdata = validation)
pv_gbm <- predict(gbmMod, newdata = validation)

#Ensemble prediction with the validation set
pv_all <- tibble(pv_glm, pv_gam, pv_lda, pv_knn, pv_rpart, pv_rf, pv_gbm) # collect the predictions of all 7 individual models
ensv_list <- apply(pv_all, 1, table) # count the groups
ensv_vote <- sapply(ensv_list, which.max) #identify the major group
pv_ens <- factor(names(ensv_vote)) #extract the group name as a factor
pv_all <- bind_cols(pv_all, tibble(pv_ens)) #add ensemble prediction to data frame of all predictions on the validation set

#Validation set accuracy using a function that cycles over all the prediction columns of the different models
accv_results <- sapply(pv_all, function(p) {round(confusionMatrix(validation$Dataset, p)$overall["Accuracy"],3)}) #

### Result
final_results <- bind_cols(acc_results, tibble(Validation = unname(accv_results))) #place the test and validation set accuracy in one data frame

#=====#