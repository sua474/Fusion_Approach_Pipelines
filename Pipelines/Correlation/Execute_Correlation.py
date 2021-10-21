import subprocess
import os


#subprocess.run(["conda","activate","plant_counting"])
execution_file = "pipeline_original.py"
input_path = "/u2/sua474/Dataset/Chiquet/"
output_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/"
filename = "Chiquet_"

for taxa in ["Taxa_30"]:
    for internal_thresold in ["10","50","100"]:
        for i in range(1,101):
            
            read_path = '{}{}/Chiquet_{}/'.format(input_path,taxa,internal_thresold)
            write_path = '{}{}/IT_{}/Correlation/'.format(output_path,taxa,internal_thresold)
        
            read_loc = "{}{}{}.csv".format(read_path,filename,str(i))
            write_loc = "{}{}{}/".format(write_path,filename,str(i))
            
            print("Processing Input File: "+filename+str(i))
            subprocess.run(["python",execution_file,"-f1",read_loc,"-m","graph_centrality","-e","kl_divergence","-min","0" ,"-c","add_one", "-st", "10" ,"-si" ,"10" ,"-cent" ,"degree" ,"-th" ,"0.5", "-cor", "pearson" ,"-cp" ,"both","-r",write_loc])

