#!/bin/bash

# ----------------------------------------------------------------------------------------------------------------------
# This file provides suggestions on how to run the simulation code on a cluster. To do so, comment out the functions
# you do not wish to run in this file and submit the job using qsub. The resulting command should look something like:
#
# qsub bash LMMPaperSim.sh
#
# However, syntax can vary between clusters so this is not guaranteed to run on your cluster without adjustment.
# ----------------------------------------------------------------------------------------------------------------------

# The below variable describes which simultion setting we are looking at. It can take the following values:
# - 1: The design for the first simulation setting
# - 2: The design for the second simulation setting
# - 3: The design for the third simulation setting
desInd=1

# The below variable specifies the output directory used by the simulations. This must be set prior to running the code.
OutDir= #...

# The below command will run 1000 simulations in serial. This may take some time... if you have access to 
# a computing cluster we suggest instead using the last command listed in this file and submitting each
# simulation as an individual job to the cluster.
python -c "from sim import LMMPaperSim; LMMPaperSim.sim2D(desInd=$desInd, OutDir='$OutDir')"

# The below commands will collate the results of all of the simulations run for the specified design into tables
# which are saved as csv files. Comment out functions if you do not need to use them.
python -c "from sim import LMMPaperSim; LMMPaperSim.differenceMetrics(desInd=$desInd, OutDir='$OutDir')"
python -c "from sim import LMMPaperSim; LMMPaperSim.performanceTables(desInd=$desInd, OutDir='$OutDir')"
python -c "from sim import LMMPaperSim; LMMPaperSim.tOutput(desInd=$desInd, OutDir='$OutDir')"

# The following command can be used to run an individual simulation. If you wish to submit each individual simlation
# to the cluster seperately you can use this command. To do this, follow the below instructions.
#
# 1: Comment out the python commands above.
# 2: Submit a job with simInd set to 1. I.e. run in the command line "qsub bash LMMPaperSim.sh 1". This will submit
#    the first simulation. This will set up the appropriate files and folders for the rest of the simulations.
# 3: Write a for loop which submits the remaining 999 simulations. Once the simulation from step 2 has finished 
#    setting things up the remaining simulations can all be run. i.e. run something like this:
#
#    ```
#    sims=1000
#    i=2
#    while [ $i -le $sims ]
#    do
#
#       qsub bash LMMPaperSim.sh $i
#       i=$(($i + 1))
#
#   done
#   ```
#
# As cluster setups can vary a lot we cannot provide exact code for the above as different systems may require
# different commands for submitting to a cluster. If the above is not clear please leave a comment on the github
# repository located at https://github.com/TomMaullin/LMMpaper.git and we will try to help as best we can.
python -c "from sim import LMMPaperSim; LMMPaperSim.runSim(simInd=$1, desInd=1, OutDir='$OutDir')"
