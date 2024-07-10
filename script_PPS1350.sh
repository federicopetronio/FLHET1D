#!/bin/bash

# Specify the filename
filename="spt100_config_a1_a2.ini" #TO CHANGE

generate_config_and_run() {
    local indiceX=$1
    local indiceY=$2

    xx=$(awk "BEGIN { printf \"%.6f\", 0.01 + $indiceX / 1000 }") # alpha1
    yy=$(awk "BEGIN { printf \"%.6f\", 0.01 + $indiceY / 1000 }") # alpha2

    # Define the output filename for each iteration
    out_filename="Results/step0_real/config_modif_cp_${indiceX}_${indiceY}.ini"
    
    # Use sed to replace the lines in the original file and save to the output file
    sed -e "s/Result dir        = Results\/zzz/Result dir        = .\/Results\/step0_real\/data_${indiceX}_${indiceY}/g" "$filename" > "$out_filename"
    sed -i "s/xxx/$xx/g" "$out_filename"
    sed -i "s/yyy/$yy/g" "$out_filename"
    
    echo "Replacement complete for indices X: $indiceX and Y: $indiceY. Modified content saved to $out_filename."
    python FLHET_compiled.py "$out_filename"
}

# Iterate over indices and launch tasks in the background
for indiceX in {-5..5}; do
    for indiceY in {-5..5}; do
        generate_config_and_run $indiceX $indiceY &
        
        # Limit the number of background jobs to avoid overloading the system
        while (( $(jobs | wc -l) >= 5 )); do
            sleep 1
        done
    done
done

# Wait for all background jobs to finish
wait
