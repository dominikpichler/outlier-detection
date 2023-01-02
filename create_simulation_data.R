# libraries we need 
library(MASS)
library(interactions)

## simulate the first dataset
set.seed(1420)
x1 = runif(n = 1500, min = 1, max = 5)
x2 = runif(n = 1500, min = 1, max = 5)

# correlation between the two 
p <- 0.075
U  <- rbinom(n=1500,1,p)

# now get the X3 that correlates with X1
x3  <- x1*U + x2*(1-U)
cor(x1, x3)

# get them to predict Y
y <- 2.46960 + (-0.26*x1) + (0.44*x3) + (0.077*x1*x3) +rnorm(250, 0, sqrt(1 - (0.0676 + 0.1936 + 0.005929)))

## bind them together
data <- data.frame(Depression = y, Neuroticism =x1, LifeStress = x3)

### run regressions 
lm1 <- lm(Depression~Neuroticism*LifeStress, data = data )
summary(lm1)

# check the interaction
interact_plot(lm1, pred = Neuroticism, modx = LifeStress, 
              modx.labels = c("Low", "Mean (0.00)", "High"), 
              color.class = c("darkgray","darkgray","#567001"),
              line.thickness = 1.3)

# check which levels are significant
sim_slopes(lm1, pred = Neuroticism, modx = LifeStress)

# get a different dataset with the slope for low levels
data2 <- data
data2$LifeStress_Categotic <- ifelse(data2$LifeStress<=(mean(data2$LifeStress)-sd(data2$LifeStress)), "YES", "NO")


## add them errors
######## now create a function to add mahalanobis outliers to a specific data
sim_outliers <- function(N, mu, Sigma, MD) {
  
  n <- length(mu)
  mu1 <- 0*mu
  
  # 
  L <- chol(Sigma)
  T <- diag(Sigma)
  Lambda <- diag(T)%*%t(L)
  Y <- matrix(0,N,n)
  
  for (k in 1:N){
    u <- mvrnorm(1, mu1, Sigma)
    u <- Lambda%*%u
    c <- t(mu1)%*%solve(Sigma)%*%mu1-MD[k]**2
    b <- t(mu1)%*%solve(Sigma)%*%u
    a <- t(u)%*%solve(Sigma)%*%u
    root <- (-b+sqrt(b**2-4*a*c))/(2*a)
    Y[k,] <- root[1]*u
  } 
  Y <- as.data.frame(Y + sample(mu, N, replace=TRUE))
  return(Y)
}

### 
### EXAMPLE ###
N <- unname(nrow(data2[data2$LifeStress_Categotic=="YES",]))

Sigma <- unname(matrix(cov(data2[data2$LifeStress_Categotic=="YES",1:2]), 2,2))
Sigma[1,1] <- 1
Sigma[2,2] <- 1
mu <- unname(colMeans(data2[data2$LifeStress_Categotic=="YES",1:2]))
MD <- rep(10,N) ## these are the Mahalanobis' distances

# get the added outliers as a data frame
x <- sim_outliers(N, mu, Sigma, MD)

# change the colnames
colnames(x) <- colnames(data2)[1:2]

# REPLACE DATA
data2[data2$LifeStress_Categotic=="YES",1:2] <- x

# DO THE MAH DIS with and without moderator
data2$mahal1 <- mahalanobis(data2[,1:2], colMeans(data2[,1:2]), cov(data2[,1:2]))

#create new column in x frame to hold p-value for each Mahalanobis distance (.001)
data2$p1 <- pchisq(data2$mahal1, df=1, lower.tail=FALSE)

# with mode
# DO THE MAH DIS with and without moderator
data2$mahal2 <- mahalanobis(data2[,1:3], colMeans(data2[,1:3]), cov(data2[,1:3]))

#create new column in x frame to hold p-value for each Mahalanobis distance
data2$p2 <- pchisq(data2$mahal2, df=2, lower.tail=FALSE)

# Typically a p-value that is less than .001 is considered to be an outlier.
dataOut <- data2[data2$p1 < 0.001,]
