# library
library(ggplot2)

# This file contains code for making the boxplot figure using simulation results. To begin,
# the below output directories must be set to folders containing `dfTable.csv` files (see
# `results/TStatisticSimulations.ipynb` for more information on generating these files).

outDir1 <- # Folder containing `dfTable.csv` for simulation setting 1...
outDir2 <- # Folder containing `dfTable.csv` for simulation setting 2...
outDir3 <- # Folder containing `dfTable.csv` for simulation setting 3...

# ------------------------------------------------------------------------------------------
# Simulation 1
# ------------------------------------------------------------------------------------------

# Read simulation results
sim1res <- read.csv(file = paste(outDir1,'/dfTable.csv',sep=''),sep=',')

# Work out true value
sim1Truth <- mean(sim1res[['Truth']])

# Read in FS results and rename columns
FSsim1 <- cbind('FS',sim1res['FS'])
FSsim1 <- rename(FSsim1, c('FS'='Degrees of Freedom Estimate', '"FS"'='Estimation Method'))

# Read in lmerTest results and rename columns
Lmertestsim1 <- cbind('lmerTest',sim1res['lmer'])
Lmertestsim1 <- rename(Lmertestsim1, c('lmer'='Degrees of Freedom Estimate', '"lmerTest"'='Estimation Method'))

# Add simulation number
sim1res <- rbind(FSsim1, Lmertestsim1)
sim1res['Simulation'] <- 'Simulation 1'

# Add truth
sim1res['Truth'] <- sim1Truth

# ------------------------------------------------------------------------------------------
# Simulation 2
# ------------------------------------------------------------------------------------------

# Read simulation results
sim2res <- read.csv(file = paste(outDir2,'dfTable.csv',sep=''),sep=',')

# Work out true value
sim2Truth <- mean(sim2res[['Truth']])

# Read in FS results and rename columns
FSsim2 <- cbind('FS',sim2res['FS'])
FSsim2 <- rename(FSsim2, c('FS'='Degrees of Freedom Estimate', '"FS"'='Estimation Method'))

# Read in lmerTest results and rename columns
Lmertestsim2 <- cbind('lmerTest',sim2res['lmer'])
Lmertestsim2 <- rename(Lmertestsim2, c('lmer'='Degrees of Freedom Estimate', '"lmerTest"'='Estimation Method'))

# Add simulation number
sim2res <- rbind(FSsim2, Lmertestsim2)
sim2res['Simulation'] <- 'Simulation 2'

# Add truth
sim2res['Truth'] <- sim2Truth

# ------------------------------------------------------------------------------------------
# Simulation 3
# ------------------------------------------------------------------------------------------

# Read simulation results
sim3res <- read.csv(file = paste(outDir3,'/Sim3/dfTable.csv',sep=''),sep=',')

# Work out true value
sim3Truth <- mean(sim3res[['Truth']])

# Read in FS results and rename columns
FSsim3 <- cbind('FS',sim3res['FS'])
FSsim3 <- rename(FSsim3, c('FS'='Degrees of Freedom Estimate', '"FS"'='Estimation Method'))

# Read in lmerTest results and rename columns
Lmertestsim3 <- cbind('lmerTest',sim3res['lmer'])
Lmertestsim3 <- rename(Lmertestsim3, c('lmer'='Degrees of Freedom Estimate', '"lmerTest"'='Estimation Method'))

# Add simulation number
sim3res <- rbind(FSsim3, Lmertestsim3)
sim3res['Simulation'] <- 'Simulation 3'

# Add truth
sim3res['Truth'] <- sim3Truth

# ------------------------------------------------------------------------------------------
# Combine Simulations
# ------------------------------------------------------------------------------------------

combinedSims <- rbind(sim1res, sim2res, sim3res)

# ------------------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------------------

# create black and white version of the plot
plot <- ggplot(combinedSims, aes(x=Simulation, y=`Degrees of Freedom Estimate`, fill=`Estimation Method`)) + geom_boxplot() + 
  scale_fill_grey(start = 0.5) + 
  geom_crossbar(data=combinedSims,aes(x = Simulation,y=Truth,ymin=Truth,
                                      ymax=Truth,fill=`Estimation Method`),
                show.legend=FALSE,position=position_dodge(),color="Black",lwd=0.3) +
  facet_wrap(~Simulation, scale="free")

# Show black and white version of the plot
print(plot)

# Creat color version of the plot
plot <- ggplot(combinedSims, aes(x=Simulation, y=`Degrees of Freedom Estimate`, fill=`Estimation Method`)) + geom_boxplot() + 
  geom_crossbar(data=combinedSims,aes(x = Simulation,y=Truth,ymin=Truth,
                                      ymax=Truth,fill=`Estimation Method`),
                show.legend=FALSE,position=position_dodge(),color="Red",lwd=0.3) +
  facet_wrap(~Simulation, scale="free")

# Show color version of the plot
print(plot)