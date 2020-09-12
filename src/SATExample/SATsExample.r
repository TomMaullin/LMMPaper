library(haven)
library(lme4)
library(tictoc)

# Path to LMMPaper directory (must be filled in)
LMMdir <- #...

# Read in reduced dataset
reducedat <- as.data.frame(read.csv(paste(LMMdir, Platform$file.sep, 'data', .Platform$file.sep, 'SATExample', .Platform$file.sep, 'school67.csv', sep="")))

# Work out response and factors
y <- as.matrix(reducedat$math)
tchrfac <- as.factor(reducedat$tchrid)
studfac <- as.factor(reducedat$studid)

# Work out design
x1 <- as.matrix(reducedat$year) 

# Run model
m1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE)
devfun1 <- lmer(y ~ x1 + (1|tchrfac) + (1|studfac),REML=FALSE,devFunOnly = TRUE)

# Time model
tic('lmer time')
tmp <- optimizeLmer(devfun1)
t <- toc()

# Fixed effects and fixed effects variances
fixef(m1)
summary(m1)$coef[, 2, drop = FALSE]

# RFX variances and residual variance
as.matrix(Matrix::bdiag(VarCorr(m1)))
resid <- as.data.frame(VarCorr(m1))[3,4]#as.data.frame(vc,order="lower.tri")

# Log-likelihood 
logLik(m1)