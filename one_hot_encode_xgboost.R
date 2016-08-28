#
# Much is borrowed from:
# https://www.kaggle.com/yibochen/talkingdata-mobile-user-demographics/xgboost-in-r-2-27217
#

# Initialize and destroy everything:
  options(stringsAsFactors=F,scipen=99)
  rm(list=ls());
  setwd("C:/Temp/201610 Kaggle TalkingData/scripts")
  gc() ; Sys.time() ; start_time <- Sys.time()
  
# Libraries
  require(data.table)
  require(Matrix)
  require(xgboost)

# Read Test and Train IDs and targets:
  temp_train <- fread("../input/gender_age_train.csv",
                     colClasses=c("character","character",
                                  "integer","character"))
  temp_test <- fread("../input/gender_age_test.csv",
                    colClasses=c("character"))

  
# Create a single object with all test and train data:
  temp_test$gender <- temp_test$age <- temp_test$group <- NA
  train_test <- rbind(temp_train,temp_test)
  setkey(train_test,device_id)
  rm(temp_test,temp_train);gc()

# Read Phone Brand data:
  brand <- fread("../input/phone_brand_device_model.csv",
               colClasses=c("character","character","character"))
  setkey(brand,device_id)
  
  # Deal with duplicate phones by device_id
    brand0 <- unique(brand,by=NULL)
    brand0 <- brand0[sample(nrow(brand0)),]
    brand2 <- brand0[-which(duplicated(brand0$device_id)),]
    train_test1 <- merge(train_test,brand2,by="device_id",all.x=T)
    rm(brand,brand0,brand2);gc()

# Read event data (note timestamp conversion)
  events <- fread("../input/events.csv",
                colClasses=c("character","character","POSIXct",
                             "numeric","numeric"))
  setkeyv(events,c("device_id","event_id"))
  events$timestamp <- as.POSIXct(events$timestamp)

# And Application Data
  event_app <- fread("../input/app_events.csv",
                   colClasses=rep("character",4))
  setkey(event_app,event_id)
  
# And Label Data:
  app_label <- fread("../input/app_labels.csv",
                     colClasses=rep("character",2))

  print("Finished Reading files")
  gc() ; Sys.time() - start_time

# There are no duplicate events... and let's keep all of our columns...
#  x <- unique(events[,list(device_id, event_id)], by=NULL)
#  which(duplicated(events[,list(device_id, event_id)]))

# Wicked data.table fu to create a comma delimited list of app_ids (apps) by event_id:
  event_apps <- event_app[,list(apps=paste(unique(app_id),collapse=",")),by="event_id"]
  device_event_apps <- merge(events,event_apps,by="event_id")

  gc() ; Sys.time() - start_time
  
# More good fu to get apps (currently indexed by event) indexed by device_id:
  f_split_paste <- function(z){paste(unique(unlist(strsplit(z,","))),collapse=",")}
  device_apps <- device_event_apps[,list(apps=f_split_paste(apps)),by="device_id"]

  tmp <- strsplit(device_apps$apps,",")
  device_apps <- data.table(device_id=rep(device_apps$device_id,
                            times=sapply(tmp,length)),
                            app_id=unlist(tmp))

  rm(device_event_apps, tmp)
  
  gc() ; Sys.time() - start_time

# Once more, with labels (a bit tricker with the extra merge)
  app_labels <- app_label[,list(labels=paste(unique(label_id),collapse=",")),by="app_id"]
  event_app_labels <- merge(event_app[,list(event_id, app_id)], app_labels, by="app_id")
  event_labels <- event_app_labels[,list(labels=f_split_paste(labels)),by="event_id"]
  device_event_labels <- merge(events[,list(event_id, device_id)], event_labels, by="event_id")
  
  device_event_labels_u <- unique(device_event_labels[,list(device_id, labels)])
  
  tmp <- strsplit(device_event_labels_u$labels,",")
  device_labels <- unique(data.table(device_id=rep(device_event_labels_u$device_id,
                                            times=sapply(tmp,length)),
                                            label_id=unlist(tmp))
                          )
  
  rm(events,event_app,event_apps)  # Can we do any of this earlier?
  rm(device_event_labels, device_event_labels_u)
  rm(app_label, app_labels, event_app_labels, event_labels)
  rm(tmp)
  
  gc() ; Sys.time() - start_time
  

# 
# Rebuild the device/feature dataframes into a single frame with a device_id column
# And a feature column containing "feature:<value>" combinations (phone_brand:OPPO, app_id:543880124725657021)
#     this guarantees a unique feature for each app, brand, model, and label
# Then lump them all together (rbind).
#
  d1 <- train_test1[,list(device_id,phone_brand)]
  train_test1$phone_brand <- NULL
  d2 <- train_test1[,list(device_id,device_model)]
  train_test1$device_model <- NULL
  d3 <- device_apps
  
  d4 <- device_labels
  
  d1[,phone_brand:=paste0("phone_brand:",phone_brand)]
  d2[,device_model:=paste0("device_model:",device_model)]
  d3[,app_id:=paste0("app_id:",app_id)]
  d4[,label_id:=paste0("label_id:",label_id)]
  names(d1) <- names(d2) <- names(d3) <- names(d4) <- c("device_id","feature_name")
  dd <- rbind(d1,d2,d3,d4)
  
  rm(device_apps, device_labels);
  rm(d1,d2,d3,d4);

  # Here's something stupid:  there are events with devices that aren't in test OR train:
  dd <- dd[device_id %in% train_test$device_id]
    
  gc() ; Sys.time() - start_time

#
# Setup a sparse matrix with rows for each device_id and columns for each feature
#
  ii <- unique(dd$device_id)          # set up a list (index locations) of all device_ids
  jj <- unique(dd$feature_name)       # and a list (index locations) of all features
  id_i <- match(dd$device_id,ii)      # get the row index for each matched device_id
  id_j <- match(dd$feature_name,jj)   # and the column index for each matched feature
  id_ij <- cbind(id_i,id_j)           # combine a big matrix of the x/y pairs for all matched features

  M <- Matrix(0,nrow=length(ii),ncol=length(jj),
              dimnames=list(ii,jj),sparse=T)  # Make a big sparse empty matrix
                                      # Note that the rownames and columnames are device and feature names respectively
  
  M[id_ij] <- 1                       # Assign 1 for every x/y pair that appears in the data set
  
  rm(ii,jj,id_i,id_j,id_ij,dd)
  
  gc() ; Sys.time() - start_time
  

#
# Separate train and test again:
#
  # x <- M[rownames(M) %in% train_test1$device_id,]  # Fixed with dd %in% above; was fixing errant device_ids
  id <- train_test1$device_id[match(rownames(M),train_test1$device_id)]  # List of device_ids in Matrix order
  y <- train_test1$group[match(rownames(M),train_test1$device_id)]  # List of groups (target) in Matrix order
  # rm(M,train_test1)


#
#  Column Reduction:
#  Only include columns that actually have values in train that aren't ALL 1
#  (It's worth noting that the second condition never seems to happen):
#
  x_train <- M[!is.na(y),]     # Select rows with valid targets; i.e. training rows
  tmp_cnt_train <- colSums(x_train)   # Number of 1s for each column in training
  M <- M[,tmp_cnt_train>0 & tmp_cnt_train<nrow(x_train)]  # Include only columns with varied data
  rm(x_train,tmp_cnt_train)


#
# Set everything up as needed for xgboost...
#
  group_name <- sort(na.omit(unique(y)))   # List of target groups
  idx_train <- which(!is.na(y))      # List of train rows in Matrix
  idx_test <- which(is.na(y))        # List of test rows in Matrix
  train_data <- M[idx_train,]        # Training features
  test_data <- M[idx_test,]          # Test features
  train <- match(y[idx_train],group_name)-1   # Ordinal target labels, required for xgboost multi: models
  test <- match(y[idx_test],group_name)-1     # These are all blank... test has no target labels yet

  valid_idx <- sample(nrow(train_data), 0.25*nrow(train_data))  # 50% validation!  Cause Overfitting!
  
  dtrain <- xgb.DMatrix(train_data[-valid_idx,],label=train[-valid_idx],missing=NA)
  dvalid <- xgb.DMatrix(train_data[valid_idx,], label=train[valid_idx],missing=NA)
  dtest <- xgb.DMatrix(test_data,label=test,missing=NA)

  param <- list(booster="gblinear",
              num_class=length(group_name),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=0.02,  # Big effect
              lambda=6,
              lambda_bias=0,
              alpha=2.2)
  
  watchlist <- list(eval=dvalid, train=dtrain)
  # set.seed(114)
  # fit_cv <- xgb.cv(params=param,
  #                  data=dtrain,
  #                  nrounds=1000,
  #                  watchlist=watchlist,
  #                  nfold=5,
  #                  early.stop.round=3,
  #                  verbose=1)
  
  ntree <- 3000
  set.seed(1729)
  fit_xgb_linear <- xgb.train(params=param,
                            data=dtrain,
                            nrounds=ntree,
                            watchlist=watchlist,
                            verbose=1,
                            print.every.n = 20,
                            early.stop.round = 20)
  
  param <- list(# booster="gblinear",
                num_class=length(group_name),
                objective="multi:softprob",
                eval_metric="mlogloss",
                max_depth = 20,
                min_child_weight = 200,
                eta=0.03,
                lambda=0,
                lambda_bias=0,
                alpha=0,
                subsample = 0.6,
                colsample = 0.6,
                early.stop.round = 10)

  watchlist <- list(eval=dvalid, train=dtrain)
# set.seed(114)
# fit_cv <- xgb.cv(params=param,
#                  data=dtrain,
#                  nrounds=100000,
#                  watchlist=watchlist,
#                  nfold=5,
#                  early.stop.round=3,
#                  verbose=1)

ntree <- 300
set.seed(1729)
fit_xgb_tree <- xgb.train(params=param,
                     data=dtrain,
                     nrounds=ntree,
                     watchlist=watchlist,
                     verbose=1,
                     print.every.n = 20)
# Look at importance (meaningless without column names, but still):
  xgb.dump(model = fit_xgb, with.stats = T, fname = 'xgb.dump')
  x <- xgb.importance(fit_xgb, feature_names = colnames(M), filename_dump = 'xgb.dump')
  x[order(x$Weight, decreasing=TRUE),]

train_pred <- predict(fit_xgb, dtrain)
train_pred_detail <- t(matrix(train_pred,nrow=length(group_name)))
res_train <- cbind(id=id[idx_train],as.data.frame(train_pred_detail))
colnames(res_train) <- c("device_id",group_name)
file_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
train_filename <- paste('../Interim/one_hot_encode_xgboost_borrowed_train_', file_timestamp, '.csv', sep="")
test_filename <- paste('../one_hot_encode_xgboost_borrowed_submit_', file_timestamp, '.csv', sep="")
write.csv(res_train, file=train_filename, row.names = F, quote = F)

test_pred <- predict(fit_xgb,dtest)
test_pred_detail <- t(matrix(test_pred,nrow=length(group_name)))
res_submit <- cbind(id=id[idx_test],as.data.frame(test_pred_detail))
colnames(res_submit) <- c("device_id",group_name)
write.csv(res_submit,file=test_filename,row.names=F,quote=F)


