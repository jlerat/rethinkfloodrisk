#!/bin/bash -l

FROOT=$NFRSK

set -a 

# Env variables
if [ -z "$1" ]
then
    USER_SUPPLIED_ARRAYS="X"
else
    USER_SUPPLIED_ARRAYS=$1
fi

# Job config
jobscript=$FROOT/scripts/fit/job_script.job

if [ ! -f "$jobscript" ]; then
    echo "ERROR - script $jobscript does not exists"
    exit 1
fi

declare -A JOBCONFIGS

# Copula fitting job
JOBCONFIGS[0,0] = "copulafit"  # Job name
JOBCONFIGS[0,1] = "10"         # number of cpus
JOBCONFIGS[0,2] = "0-5"        # arrays
JOBCONFIGS[0,3] = "1"          # use user supplied array if any
JOBCONFIGS[0,4] = "X"          # job dependency
JOBCONFIGS[0,5] = "X"          # job id

# General postprocessing job
JOBCONFIGS[1,0] = "postprocess"  # Job name
JOBCONFIGS[1,1] = "1"            # number of cpus
JOBCONFIGS[1,2] = "0-5"          # arrays
JOBCONFIGS[1,3] = "1"            # use user supplied array if any
JOBCONFIGS[1,4] = "0"            # parent job number
JOBCONFIGS[1,5] = "X"            # job id

# MVN postprocessing job
JOBCONFIGS[2,0] = "mvnprocess"   # Job name
JOBCONFIGS[2,1] = "1"            # number of cpus
JOBCONFIGS[2,2] = "0-799"        # arrays
JOBCONFIGS[2,3] = "0"            # use user supplied array if any
JOBCONFIGS[2,4] = "0"            # parent job number
JOBCONFIGS[2,5] = "X"            # job id

NJOBS=3

for ((ijob = 0; ijob < $NJOBS; ijob ++)); do
    # Retrieve config
    jobname=$JOBCONFIGS[$ijob,0]
    ncpus=$JOBCONFIGS[$ijob,1]
    arrays=$JOBCONFIGS[$ijob,2]
    overwrite_arrays=$JOBCONFIGS[$ijob,3]
    parent_job=$JOBCONFIGS[$ijob,4]

    # Overwrite array with user supplied argument
    if [ $overwrite_arrays eq "1"] && [ $USER_SUPPLIED_ARRAYS -ne "X"]
    then
        arrays=$USER_SUPPLIED_ARRAYS        
    fi
    
    # Get parent job id
    if [ $parent_job -neq "X" ] 
    then
        parent_ijob=$((parent_job))
        parent_jobid=$JOBCONFIGS[$parent_ijob,5]
    else    
        parent_jobid="X"
    fi

    echo
    echo ----------------------------------
    echo JOB \#$ijob
    echo   NAME     : $jobname
    echo   NCPUS    : $ncpus
    echo   ARRAYS   : $arrays
    echo   PJOBID   : $parent_jobid
    
    # Create log folder
    FLOG=$FROOT/logs/$jobname
    mkdir -p $FLOG
    echo .. created log folder logs/$jobname

    # Run job
    if [ $parent_jobid -ne "X" ]
    then
        echo .. submitting job with array $arrays and no dependency
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$arrays --parsable --export=ALL $jobscript)
    else    
        echo .. submitting job with array $arrays and dependency on $parent_jobid
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$arrays --dependency=afterany:${parent_jobid} --export=ALL $jobscript)
    fi    

    # Iterate jobid
    JOBCONFIGS[$ijob,5]=$jobid

    echo .. All done. Submitted job $jobid and moving to next job.
    echo ----------------------------------
    echo

done    


