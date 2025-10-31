run_mod=$(sbatch LEC_only_run.sbatch)
run_mod_id=$(echo $run_mod | awk '{print $4}')
run_an_output=$(sbatch --dependency=aftercorr:$run_mod_id lec_analysis.sbatch)
