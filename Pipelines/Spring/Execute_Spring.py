import subprocess
import os


#subprocess.run(["conda","activate","plant_counting"])
execution_file = "Implementation.R"
input_path = "/u2/sua474/Dataset/Chiquet/"
output_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/"
filename = "Chiquet_"

#Params
nlambda = 15

for taxa in ["Taxa_30"]:
    for internal_thresold in ["10","50","100"]:
        for i in range(1,101):
            read_path = '{}{}/Chiquet_{}/'.format(input_path,taxa,internal_thresold)
            write_path = '{}{}/IT_{}/Spring/'.format(output_path,taxa,internal_thresold)
        
            read_loc = "{}{}{}.csv".format(read_path,filename,str(i))
            write_loc = "{}{}{}/".format(write_path,filename,str(i))
        
            print("Processing Input File: "+filename+str(i)+" Taxa: "+str(taxa)+" IT: "+str(internal_thresold)+"\n")
            subprocess.run(["Rscript",execution_file,read_loc,write_loc,str(nlambda)])

