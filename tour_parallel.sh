#!/bin/bash

# Specify the filename
filename="SPT100_gradP/spt100_config_alpha_cst.ini"

generate_config_and_run() {
    local indiceX=$1
    local indiceY=$2

    xx=$(awk "BEGIN { printf \"%.6f\", 0.001 + 0.001 * $indiceX }")
    yy=$(awk "BEGIN { printf \"%.6f\", 0.004 + 0.001 * $indiceY }")

    # Define the output filename for each iteration
    out_filename="SPT100_gradP/alpha_B_cst1/config_modif_cp_${indiceX}_${indiceY}.ini"
    
    # Use sed to replace the lines in the original file and save to the output file
    sed -e "s/Result dir        = \.\/SPT100_gradP\/alpha_B_cst1\/zzz/Result dir        = .\/SPT100_gradP\/alpha_B_cst1\/data_cp_${indiceX}_${indiceY}/g" "$filename" > "$out_filename"
    sed -i "s/xxx/$xx/g" "$out_filename"
    sed -i "s/yyy/$yy/g" "$out_filename"
    
    echo "Replacement complete for indices X: $indiceX, Y: $indiceY. Modified content saved to $out_filename."
    python FLHET_compiled_gradP.py "$out_filename"
}

# Iterate over indices and launch tasks in the background
for indiceX in {1..10}; do
    for indiceY in {1..10}; do
        generate_config_and_run $indiceX $indiceY &
        
        # Limit the number of background jobs to avoid overloading the system
        while (( $(jobs | wc -l) >= 10 )); do
            sleep 1
        done
    done
done

# Wait for all background jobs to finish
wait
