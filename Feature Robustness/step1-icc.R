rm(list = ls())
library(readr)
library(irr)

feature_1 <- read.csv('Original.csv',header=TRUE,check.names = F,row.names = 1)
feature_2 <- read.csv('IntraICC.csv',header=TRUE,check.names = F,row.names = 1)
feature_3 <- read.csv('InterICC.csv',header=TRUE,check.names = F,row.names = 1)
len <- 1130# N是指标的数量
icc_val<-vector(length=len)
thr <- 0.8
#第一次进行组内ICC
for (i in 1:len){#len是特征的数量，第一列是ID，从第二列开始进行指标的比较
  ratings <- cbind(feature_1[,i],feature_2[,i])
  icc <- icc(ratings, model = "twoway", 
             type = "agreement", 
             unit = "single", r0 = 0, conf.level = 0.95)
  icc_val[i] <- icc$value
}

Index <- which(icc_val > thr)
mean(icc_val)
feature_intra=feature_1[,Index]
#第二次进行组间ICC
for (i in 1:len){#len是特征的数量，第一列是ID，从第二列开始进行指标的比较
  ratings <- cbind(feature_1[,i],feature_3[,i])
  icc <- icc(ratings, model = "twoway", 
             type = "agreement", 
             unit = "single", r0 = 0, conf.level = 0.95)
  icc_val[i] <- icc$value
}

Index <- which(icc_val > thr)
mean(icc_val)
feature_inter=feature_1[,Index]

table(colnames(feature_intra) %in% colnames(feature_inter))
table(colnames(feature_inter) %in% colnames(feature_intra))


feature_last=feature_inter[,colnames(feature_inter) %in% colnames(feature_intra)]
feature= read.csv('Features_Radiomics.csv',header = T,check.names = F,row.names = 1)
feature_icc=feature[,colnames(feature) %in% colnames(feature_last)]
write.csv(feature_icc,file = 'feature_icc.csv',row.names=TRUE)


