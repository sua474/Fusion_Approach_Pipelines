import subprocess
import os


#subprocess.run(["conda","activate","plant_counting"])
execution_file = "Implementation.R"
input_path = "/u2/sua474/Dataset/Chiquet/Taxa_10/Chiquet_10/"
output_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/Taxa_10/IT_10/Spring/"
filename = "Chiquet_"

#Params
nlambda = 15

for i in range(1,2):
    read_loc = "{}{}{}.csv".format(input_path,filename,str(i))
    write_loc = "{}{}{}/".format(output_path,filename,str(i))
    print("Processing Input File: "+filename+str(i))
    subprocess.run(["Rscript",execution_file,read_loc,write_loc,str(nlambda)])

