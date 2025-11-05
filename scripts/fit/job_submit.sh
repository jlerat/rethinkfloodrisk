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
    if [ $ARRAYS = "none" ]
    then
        if [ $jobidprev = -999 ]
        then
            echo ... submitting job with no array and no dependency
            jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
                --parsable --export=ALL $jobscript)
        else
            echo ... submitting job with no array and dependency on $jobidprev
            sbatch -J $jobname --cpus-per-task=$ncpus \
                --dependency=afterany:${jobidprev} --export=ALL $jobscript
        fi    
    else
        if [ $jobidprev = -999 ]
        then
            echo ... submitting job with array $ARRAYS and no dependency
            jobid=$(sbatch -J $jobname --cpus-per-task=$ncpus \
                --array=$ARRAYS --parsable --export=ALL $jobscript)
        else    
            echo ... submitting job with array $ARRAYS and dependency on $jobidprev
            sbatch -J $jobname --cpus-per-task=$ncpus \
                --array=$ARRAYS --dependency=afterany:${jobidprev} --export=ALL $jobscript
        fi    
    fi

    # Iterate jobid
    jobidprev=$jobid

    echo ... All done. Moving to next job.
done    


