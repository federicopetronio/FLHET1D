#!/bin/bash

# Choose the thruster and the model ###TO CHANGE###

thruster="PPSX00K"
model="Step"

# Specify the filename
filename="config_"$thruster"_1D_mazallon.ini"

generate_config_and_run() {

    local indiceX=$1
    local indiceY=$2 

    #Define the alpha_B1 and alpha_B2 values ###TO CHANGE###
    alphaB1=$(awk "BEGIN { printf \"%.6f\", 0.010 + $indiceX / 1000 }")  
    alphaB2=$(awk "BEGIN { printf \"%.6f\", 0.010 + $indiceY / 1000 }") 
    # Define the output filename for each iteration4
    out_filename="Results_1D/$thruster/$model/config_modif_cp_${indiceX}_${indiceY}.ini"
    
    # Use sed to replace the lines in the original file and save to the output file
    sed -e "s/Result dir        = Results_1D\/dest/Result dir        = .\/Results_1D\/$thruster\/$model\/data_${indiceX}_${indiceY}/g" "$filename" > "$out_filename"
    sed -i "s/al1/$alphaB1/g" "$out_filename"
    sed -i "s/al2/$alphaB2/g" "$out_filename"
    sed -i "s/case/$model/g" "$out_filename"
    
    echo "Replacement complete for indices X: $indiceX and Y: $indiceY. Modified content saved to $out_filename."
    python FLHET_1D_compiled_mazallon.py "$out_filename" 
}

# Iterate over indices and launch tasks in the background ###TO CHANGE###
for indiceX in {-5..5}; do
    for indiceY in {-5..5}; do
        generate_config_and_run $indiceX $indiceY &
        
        # Limit the number of background jobs to avoid overloading the system
        while (( $(jobs | wc -l) >= 11 )); do
            sleep 1
        done
    done
done

# Wait for all background jobs to finish
wait
