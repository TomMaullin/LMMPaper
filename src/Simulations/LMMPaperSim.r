#!/apps/well/R/3.4.3/bin/Rscript
#$ -cwd
#$ -q short.qc
#$ -o ./loglmer/
#$ -e ./loglmer/

library(MASS)
library(Matrix)
library(lme4)
library(tictoc)

# ---------------------------------------------------------------------------------------
# IMPORTANT: Input options
# ---------------------------------------------------------------------------------------
#
# The below variables control which simulation is run and how. The variable names match
# those used in the `LMMPaperSim.py` file and are given as follows:
#
# - OutDir: The output directory.
# - desInd: Integer value between 1 and 3 representing which design to run. The 
#           designs are as follows:
#           - Design 1: nlevels=[50], nraneffs=[2]
#           - Design 2: nlevels=[50,10], nraneffs=[3,2]
#           - Design 3: nlevels=[100,50,10], nraneffs=[4,3,2]
# - nsim: Number of simulations (default=1000)
# - mode: String indicating whether to run parameter estimation simulations (mode=
#         'param') or T statistic simulations (mode='Tstat').
# - reml: Boolean indicating whether to use ML or ReML estimation. 
#
# ---------------------------------------------------------------------------------------
# For parameter simulations example use this directory
outDir <- paste(dirname(rstudioapi::getSourceEditorContext()$path), '..', '..', 'data', 'ParamSimulation', sep=.Platform$file.sep)
# For T simulations example use this directory
outDir <- paste(dirname(rstudioapi::getSourceEditorContext()$path), '..', '..', 'data', 'TstatSimulation', sep=.Platform$file.sep)
desInd <- 2
nsim <- 3
mode <- 'Tstat'
reml <- FALSE
# ---------------------------------------------------------------------------------------

# If we are timing code, don't import lmerTest since it can reduces the performance of lmer
if (mode=='param'){
  timing <- TRUE
} else {
  timing <- FALSE  
}
if (!timing){
  library(lmerTest)
}

# Loop through the simulations running lmer
for (simInd in 1:nsim){
  if (desInd==3){
    
    results <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_results.csv',sep=''))
    
    X <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_X.csv',sep=''),sep=' ', header=FALSE)
    Y <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Y.csv',sep=''),sep=' ', header=FALSE)
    Zfactor0 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zfactor0.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata0 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zdata0.csv',sep=''),sep=' ', header=FALSE)
    Zfactor1 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zfactor1.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata1 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zdata1.csv',sep=''),sep=' ', header=FALSE)
    Zfactor2 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zfactor2.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata2 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design3_Zdata2.csv',sep=''),sep=' ', header=FALSE)
    y <- as.matrix(Y[,1])
    
    x2 <- as.matrix(X[,2])
    x3 <- as.matrix(X[,3])
    x4 <- as.matrix(X[,4])
    x5 <- as.matrix(X[,5])
    
    
    z01 <- as.matrix(Zdata0[,1])
    z02 <- as.matrix(Zdata0[,2])
    z03 <- as.matrix(Zdata0[,3])
    z04 <- as.matrix(Zdata0[,4])
    
    z11 <- as.matrix(Zdata1[,1])
    z12 <- as.matrix(Zdata1[,2])
    z13 <- as.matrix(Zdata1[,3])
    
    z21 <- as.matrix(Zdata2[,1])
    z22 <- as.matrix(Zdata2[,2])
    
    m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03 + z04|Zfactor0) + (0 + z11 + z12 + z13|Zfactor1) + (0 + z21 + z22|Zfactor2), REML=reml) #Don't need intercepts in R - automatically assumed
    
    devfun <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03 + z04|Zfactor0) + (0 + z11 + z12 + z13|Zfactor1) + (0 + z21 + z22|Zfactor2), REML=reml, devFunOnly = TRUE) #Don't need intercepts in R - automatically assumed
    
    tic('lmer time 2')
    opt<-optimizeLmer(devfun)
    t<-toc()
    
    lmertime <- t$toc-t$tic
    
    
    results[1,'lmer']<-lmertime
    results[2,'lmer'] <- opt$feval
    if (!reml){
      results[3,'lmer']<-logLik(m)[1]
    } else {
      results[3,'lmer']<-logLik(m)[1]
    }
    results[4:8,'lmer'] <- fixef(m)
    results[9,'lmer']<-as.data.frame(VarCorr(m))$vcov[20]
    
    Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
    
    vechD0 <- Ds[1:4,1:4][lower.tri(Ds[1:4,1:4],diag = TRUE)]
    vechD1 <- Ds[5:7,5:7][lower.tri(Ds[5:7,5:7],diag = TRUE)]
    vechD2 <- Ds[8:9,8:9][lower.tri(Ds[8:9,8:9],diag = TRUE)]
    
    results[10:19,'lmer']<-vechD0/as.data.frame(VarCorr(m))$vcov[20]
    results[20:25,'lmer']<-vechD1/as.data.frame(VarCorr(m))$vcov[20]
    results[26:28,'lmer']<-vechD2/as.data.frame(VarCorr(m))$vcov[20]
    results[29:38,'lmer']<-vechD0
    results[39:44,'lmer']<-vechD1
    results[45:47,'lmer']<-vechD2
    
    # If we're not using the simulations to test performance, output everything we can.
    if (!timing){
      # Run T statistic inference
      Tresults<-lmerTest::contest1D(m, c(0,0,0,0,1),ddf=c("Satterthwaite"))
      p<-Tresults$`Pr(>|t|)`
      Tstat<-Tresults$`t value`
      df<-Tresults$df
      
      # Make p-values 1 sided
      if (Tstat>0){
        p <- p/2
      } else {
        p <- 1-p/2
      }
      
      results[48,'lmer']<-Tstat
      results[49,'lmer']<-p
      results[50,'lmer']<-df
    }
    
    write.csv(results,paste(outDir,'/Sim',toString(simInd),'_Design3_results.csv',sep=''), row.names = FALSE)
    
    
  } else if (desInd==2){
    
    results <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_results.csv',sep=''))
    
    X <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_X.csv',sep=''),sep=' ', header=FALSE)
    Y <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_Y.csv',sep=''),sep=' ', header=FALSE)
    Zfactor0 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_Zfactor0.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata0 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_Zdata0.csv',sep=''),sep=' ', header=FALSE)
    Zfactor1 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_Zfactor1.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata1 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design2_Zdata1.csv',sep=''),sep=' ', header=FALSE)
    
    y <- as.matrix(Y[,1])
    
    x2 <- as.matrix(X[,2])
    x3 <- as.matrix(X[,3])
    x4 <- as.matrix(X[,4])
    x5 <- as.matrix(X[,5])
    
    
    z01 <- as.matrix(Zdata0[,1])
    z02 <- as.matrix(Zdata0[,2])
    z03 <- as.matrix(Zdata0[,3])
    
    z11 <- as.matrix(Zdata1[,1])
    z12 <- as.matrix(Zdata1[,2])
    
    m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03|Zfactor0) + (0 + z11 + z12|Zfactor1), REML=reml) #Don't need intercepts in R - automatically assumed
    
    devfun <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02 + z03|Zfactor0) + (0 + z11 + z12|Zfactor1), REML=reml, devFunOnly = TRUE) #Don't need intercepts in R - automatically assumed
    
    tic('lmer time')
    opt<-optimizeLmer(devfun)
    t<-toc()
    
    lmertime <- t$toc-t$tic
    
    results[1,'lmer']<-lmertime
    results[2,'lmer'] <- opt$feval
    if (!reml){
      results[3,'lmer']<-logLik(m)[1]
    } else {
      results[3,'lmer']<-logLik(m)[1]
    }
    results[4:8,'lmer'] <- fixef(m)
    results[9,'lmer']<-as.data.frame(VarCorr(m))$vcov[10]
    
    Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
    
    vechD0 <- Ds[1:3,1:3][lower.tri(Ds[1:3,1:3],diag = TRUE)]
    vechD1 <- Ds[4:5,4:5][lower.tri(Ds[4:5,4:5],diag = TRUE)]
    
    results[10:15,'lmer']<-vechD0/as.data.frame(VarCorr(m))$vcov[10]
    results[16:18,'lmer']<-vechD1/as.data.frame(VarCorr(m))$vcov[10]
    
    results[19:24,'lmer']<-vechD0
    results[25:27,'lmer']<-vechD1
    
    # If we're not using the simulations to test performance, output everything we can.
    if (!timing){
      # Run T statistic inference
      Tresults<-lmerTest::contest1D(m, c(0,0,0,0,1),ddf=c("Satterthwaite"))
      p<-Tresults$`Pr(>|t|)`
      Tstat<-Tresults$`t value`
      df<-Tresults$df
      
      # Make p-values 1 sided
      if (Tstat>0){
        p <- p/2
      } else {
        p <- 1-p/2
      }
      
      results[28,'lmer']<-Tstat
      results[29,'lmer']<-p
      results[30,'lmer']<-df
    }
    
    write.csv(results,paste(outDir,'/Sim',toString(simInd),'_Design2_results.csv',sep=''), row.names = FALSE)
    
  } else if ((desInd==1) || (desInd==4)){
    
    results <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_results.csv',sep=''))
    
    X <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_X.csv',sep=''),sep=' ', header=FALSE)
    Y <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_Y.csv',sep=''),sep=' ', header=FALSE)
    Zfactor0 <- factor(read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_Zfactor0.csv',sep=''), sep=' ', header=FALSE)[,1])
    Zdata0 <- read.csv(file = paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_Zdata0.csv',sep=''),sep=' ', header=FALSE)
    
    y <- as.matrix(Y[,1])
    
    x2 <- as.matrix(X[,2])
    x3 <- as.matrix(X[,3])
    x4 <- as.matrix(X[,4])
    x5 <- as.matrix(X[,5])
    
    
    z01 <- as.matrix(Zdata0[,1])
    z02 <- as.matrix(Zdata0[,2])
    
    m <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02|Zfactor0), REML=reml) #Don't need intercepts in R - automatically assumed
    
    devfun <- lmer(y ~ x2 + x3 + x4 + x5 + (0 + z01 + z02|Zfactor0), REML=reml, devFunOnly = TRUE) #Don't need intercepts in R - automatically assumed
    
    tic('lmer time')
    opt<-optimizeLmer(devfun)
    t<-toc()
    
    lmertime <- t$toc-t$tic
    
    results[1,'lmer']<-lmertime
    results[2,'lmer'] <- opt$feval
    if (!reml){
      results[3,'lmer']<-logLik(m)[1]
    } else {
      results[3,'lmer']<-logLik(m)[1]
    }
    results[4:8,'lmer'] <- fixef(m)
    results[9,'lmer']<-as.data.frame(VarCorr(m))$vcov[4]
    
    Ds <- as.matrix(Matrix::bdiag(VarCorr(m)))
    
    vechD0 <- Ds[1:2,1:2][lower.tri(Ds[1:2,1:2],diag = TRUE)]
    
    results[10:12,'lmer']<-vechD0/as.data.frame(VarCorr(m))$vcov[10]
    results[13:15,'lmer']<-vechD0
    
    # If we're not using the simulations to test performance, output everything we can.
    if (!timing){
      # Run T statistic inference
      Tresults<-lmerTest::contest1D(m, c(0,0,0,0,1),ddf=c("Satterthwaite"))
      p<-Tresults$`Pr(>|t|)`
      Tstat<-Tresults$`t value`
      df<-Tresults$df
      
      # Make p-values 1 sided
      if (Tstat>0){
        p <- p/2
      } else {
        p <- 1-p/2
      }
      
      results[16,'lmer']<-Tstat
      results[17,'lmer']<-p
      results[18,'lmer']<-df
    }
    
    write.csv(results,paste(outDir,'/Sim',toString(simInd),'_Design',toString(desInd),'_results.csv',sep=''), row.names = FALSE)
    
  }
}
results