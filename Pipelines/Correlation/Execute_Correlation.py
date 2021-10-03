import subprocess
import os


#subprocess.run(["conda","activate","plant_counting"])
execution_file = "pipeline_original.py"
input_path = "/u2/sua474/Dataset/Chiquet/Taxa_10/Chiquet_10/"
output_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/Taxa_10/IT_10/Correlation/"
filename = "Chiquet_"

for i in range(1,2):
    read_loc = "{}{}{}.csv".format(input_path,filename,str(i))
    write_loc = "{}{}{}/".format(output_path,filename,str(i))
    print("Processing Input File: "+filename+str(i))
    subprocess.run(["python",execution_file,"-f1",read_loc,"-m","graph_centrality","-e","kl_divergence","-min","0" ,"-c","add_one", "-st", "10" ,"-si" ,"10" ,"-cent" ,"degree" ,"-th" ,"0.2", "-cor", "pearson" ,"-cp" ,"both","-r",write_loc])

