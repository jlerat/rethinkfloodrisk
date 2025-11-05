#!/bin/bash -l

FROOT=$NFRSK

set -a 

# Env variables
if [ -z "$1" ]
then
    ARRAYS="none"
else
    ARRAYS=$1
fi

# Job config
jobscript=$FROOT/scripts/fit/job_script.job

if [ ! -f "$jobscript" ]; then
    echo "ERROR - script $jobscript does not exists"
    exit 1
fi

job1=copulafit
job2=postprocess

echo
echo -----------------------
echo JOBNAME = $jobname
echo
echo ARRAYS = $ARRAYS
echo -----------------------
echo

# Create log folder
FLOG1=$FROOT/logs/$job1
mkdir -p $FLOG1

FLOG2=$FROOT/logs/$job2
mkdir -p $FLOG2

# Run job
if [ $ARRAYS = "none" ]
then
    job1id=$(sbatch -J $job1 --parsable --export=ALL $jobscript)
    sbatch -J $job2 --dependency=afterany:${job1id} --export=ALL $jobscript
else
    job1id=$(sbatch -J $job1 --array=$ARRAYS --parsable --export=ALL $jobscript)
    sbatch -J $job2 --array=$ARRAYS --dependency=afterany:${job1id} --export=ALL $jobscript
fi

