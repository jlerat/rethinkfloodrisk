#!/bin/bash -l

FROOT=$NFRSK

set -a 

# Env variables
VERSION=$1
if [ -z "$1" ]
then
    echo "ERROR - Expected a version number as first argument"
    exit 1
fi

if [ -z "$2" ]
then
    USER_SUPPLIED_ARRAYS="X"
else
    USER_SUPPLIED_ARRAYS=$2
fi

# Configure array numbers
NTASKS=144
ARRAYS_FIT="0-$(($NTASKS - 1))"    
ARRAYS_PROC="0-$((2*$NTASKS - 1))"
ARRAYS_MVN="0-$((5*$NTASKS - 1))"

# Job config
JOBSCRIPT=$FROOT/scripts/fit/job_script.job

if [ ! -f "$JOBSCRIPT" ]; then
    echo "ERROR - script $JOBSCRIPT does not exists"
    exit 1
fi

# Mimick a 2d array using 1d bash arrays
declare -A JOBCONFIGS

# Copula fitting job
JOBCONFIGS["0,0"]="copulafit"  # Job name
JOBCONFIGS["0,1"]="10"         # number of cpus
JOBCONFIGS["0,2"]=$ARRAYS_FIT  # arrays
JOBCONFIGS["0,3"]="1"          # use user supplied array if any
JOBCONFIGS["0,4"]="X"          # job dependency
JOBCONFIGS["0,5"]="X"          # job id

# General postprocessing job (depends on copula fitting)
JOBCONFIGS["1,0"]="postprocess"  # Job name
JOBCONFIGS["1,1"]="1"            # number of cpus
JOBCONFIGS["1,2"]=$ARRAYS_PROC   # arrays
JOBCONFIGS["1,3"]="0"            # use user supplied array if any
JOBCONFIGS["1,4"]="0"            # parent job number
JOBCONFIGS["1,5"]="X"            # job id

# MVN postprocessing job (depends on copula fitting)
JOBCONFIGS["2,0"]="mvnprocess"   # Job name
JOBCONFIGS["2,1"]="1"            # number of cpus
JOBCONFIGS["2,2"]=$ARRAYS_MVN    # arrays
JOBCONFIGS["2,3"]="0"            # use user supplied array if any
JOBCONFIGS["2,4"]="0"            # parent job number
JOBCONFIGS["2,5"]="X"            # job id

# MVN postprocessing concat job (depends on MVN postprocessing)
JOBCONFIGS["3,0"]="mvnconcat"    # Job name
JOBCONFIGS["3,1"]="1"            # number of cpus
JOBCONFIGS["3,2"]="0"            # arrays
JOBCONFIGS["3,3"]="0"            # use user supplied array if any
JOBCONFIGS["3,4"]="2"            # parent job number
JOBCONFIGS["3,5"]="X"            # job id


LENGTH=${#JOBCONFIGS[@]}
NJOBS=$(($LENGTH / 6))    

echo
echo "**************************************************"
echo Processing $NJOBS JOBS
echo User supplied arrays : $USER_SUPPLIED_ARRAYS
echo Arrays fit default  : $ARRAYS_FIT
echo Arrays process default  : $ARRAYS_PROC
echo Froot : $FROOT
echo "**************************************************"
echo

for ((ijob = 0; ijob < $NJOBS; ijob ++)); do
    # Retrieve config
    jobname=${JOBCONFIGS[$ijob,0]}
    ncpus=${JOBCONFIGS[$ijob,1]}
    arrays=${JOBCONFIGS[$ijob,2]}
    overwrite_arrays=${JOBCONFIGS[$ijob,3]}
    parent_job=${JOBCONFIGS[$ijob,4]}

    # Overwrite array with user supplied argument
    if [[ $overwrite_arrays = "1" && $USER_SUPPLIED_ARRAYS != "X" ]]
    then
        arrays=$USER_SUPPLIED_ARRAYS        
    fi
    
    # Get parent job id
    if [ $parent_job != "X" ] 
    then
        parent_ijob=$((parent_job))
        parent_jobid=${JOBCONFIGS[$parent_ijob,5]}
    else    
        parent_jobid="X"
    fi

    echo
    echo ----------------------------------------------------------
    echo JOB \#$ijob
    echo NAME      : $jobname
    echo NCPUS     : $ncpus
    echo OVERWRITE : $overwrite_arrays
    echo ARRAYS    : $arrays
    echo PJOBID    : $parent_jobid

    # Create log folder
    FLOG=$FROOT/logs/$jobname
    mkdir -p $FLOG
    echo .. created log folder logs/$jobname

    # Run job
    if [ $parent_jobid = "X" ]
    then
        echo .. submitting job with array $arrays and no dependency
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$arrays --parsable --export=ALL,VERSION $JOBSCRIPT)
    else    
        echo .. submitting job with array $arrays and dependency on $parent_jobid
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$arrays --parsable --dependency=afterany:${parent_jobid} \
            --export=ALL,VERSION $JOBSCRIPT)
    fi    

    # Iterate jobid
    JOBCONFIGS["$ijob,5"]=$jobid

    echo .. All done. Submitted job $jobid and moving to next job.
    echo ----------------------------------------------------------
    echo

done    


