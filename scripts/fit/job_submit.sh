#!/bin/bash -l

FROOT=$NFRSK

set -a 

# Env variables
if [ -z "$1" ]
then
    ARRAYS="0-5"
else
    ARRAYS=$1
fi

# Job config
jobscript=$FROOT/scripts/fit/job_script.job

if [ ! -f "$jobscript" ]; then
    echo "ERROR - script $jobscript does not exists"
    exit 1
fi

JOBNAMES=(
    copulafit
    postprocess
)   

jobidprev=-999

for jobname in "${JOBNAMES[@]}"; do
    echo
    echo ----------------------------------
    echo JOBNAME = $jobname

    if [ $jobname = "copulafit" ]
    then
        ncpus=10
    else
        ncpus=1
    fi
    echo .. ncpus = $ncpus

    # Create log folder
    FLOG=$FROOT/logs/$jobname
    mkdir -p $FLOG
    echo .. created log folder logs/$jobname

    # Run job
    if [ $jobidprev = -999 ]
    then
        echo .. submitting job with array $ARRAYS and no dependency
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$ARRAYS --parsable --export=ALL $jobscript)
    else    
        echo .. submitting job with array $ARRAYS and dependency on $jobidprev
        jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
            --array=$ARRAYS --dependency=afterany:${jobidprev} --export=ALL $jobscript)
    fi    

    # Iterate jobid
    jobidprev=$jobid

    echo .. All done. Submitted job $jobid and moving to next job.
    echo
done    


