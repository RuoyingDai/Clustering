# ############################### IMPORT PACKAGES ###############################
library("PMA");library('fastDummies');library(lme4)
if (!require("NbClust")) install.packages("NbClust")
library(NbClust)


# ######################### IMPORT COVID DATASET ################################
# EXTRACT FEATURES FROM DATASET
rawdata <- read.csv('D:/BDS/3UML/UML/feature.csv')
mX <- as.matrix(rawdata[,6:1678])
mX[is.na(mX)] <- 0
code <- as.vector(rawdata[,1])
country <- as.vector(rawdata[,2])
case <- as.vector(rawdata[,3])
death <- as.vector(rawdata[,4])
pop <- as.vector(rawdata[,5])
case_rate <- case/pop
death_rate <- death/pop


##################### k-means (my implementation)###############################

assignCluster <- function(mX, mCENTER, distance ='Euclidean'){
  vpoint_cluster <- list()
  if (distance =='Euclidean'){
    for (point in seq(1,nrow(mX))){
      old_center_distance <- 9999
      new_center <- 0
      for(k in seq(1, nrow(mCENTER))){
        new_center_distance <- dist(rbind(mX[point,], mCENTER[k,]), method = 'euclidean')
        if (new_center_distance < old_center_distance){
          old_center_distance <- new_center_distance
          new_center <- k}
      }
      vpoint_cluster <- append(vpoint_cluster, new_center)
    }
  }
  return(vpoint_cluster) # vector
}

kmeans_alt<- function(mX, k_cluster, distance = 'Euclidean'){
  # initialize variables
  row <- nrow(mX)
  j <- 0 # iteration counter
  n <- row # number of switch
  vcenter_id <- sample(1:row, k_cluster) # select random points as cluster center
  mCENTER <- mX[vcenter_id,]
  initial_mCENTER <- mCENTER
  
  # iteration
  while (n > 0 ){
    j <- j + 1
    if(j == 1){
      v_new_point_cluster <- assignCluster(mX, mCENTER, distance ='Euclidean')
      mCENTER <- matrix(nrow = k_cluster, ncol=ncol(mX))
      for (k in seq(1, k_cluster)){
        k_mean <- colMeans(mX[v_new_point_cluster ==k,])
        mCENTER[k, ] <- k_mean}
      next}
    v_old_point_cluster <- v_new_point_cluster
    v_new_point_cluster <- assignCluster(mX, mCENTER, distance ='Euclidean')
    n <- sum(unlist(v_new_point_cluster) != unlist(v_old_point_cluster))  # define output
    mCENTER <- matrix(nrow = k_cluster, ncol=ncol(mX))
    for (k in seq(1, k_cluster)){
    k_mean <- colMeans(mX[v_new_point_cluster ==k,])
    mCENTER[k, ] <- k_mean
    }
  }
  
  # record encoder and cluster mean
  encoder <- list(
    mCENTER <- mCENTER,# cluster mean
    initial_mCENTER <- initial_mCENTER
    
  )
  
  return(encoder)
}

set.seed(45)
encoder <- kmeans_alt(mX, k_cluster = 3, distance = 'Euclidean')
mX_cluster <- assignCluster(mX, as.matrix(encoder[[1]]))


###################### k-means (package) ###############################


package_result <-kmeans(mX, centers = 3, algorithm = "Lloyd",iter.max= 100) 
package_result$cluster
sum(unlist(mX_cluster)!=package_result$cluster)


# ################### hierarchical (package) ##########################
library(ggplot2)
library(ggdendro)

# GET CLUSTER NUMBER
nbHier <- NbClust(data = as.matrix(mX), distance = "euclidean", 
                  method = "complete", index = "ch")
# GET CLUSTER RESULT
result.hierarch <- hclust(dist(mX), method = "average")
clus3 = cutree(result.hierarch, 3)

# PLOT
ggdendrogram(result.hierarch)

# STATISTICS COMPUTATION
summary(death_rate[clus3==1])
summary(death_rate[clus3==2])
summary(death_rate[clus3==3])
summary(case_rate[clus3==1])
summary(case_rate[clus3==2])
summary(case_rate[clus3==3])
