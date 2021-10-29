# Fusion_Approach_Pipelines
 This repo contains different implementations (algorithms) of microbial interaction network creation. <br/>
 
1) Each folder inside the *pipeline* folder contains the default implementation of a research paper. Each folder contains
an Implementation file, Installation_Steps file, *Execute_* file and Github file referring to the source Github directory.<br/>
Each folder also contains a *script* folder that contains the .sh scripts of ececution. <br/>
2) The *Analytics_Code* folder contains the individual analytics methods written for data analytics <br/>
3) Inside the *Analytics_Code* folder there is another folder named *Consolidated_Analytics_Pipeline*. This folder has the latest consolidated data analytics pipeline. <br/> 

**Note: Xiao_and_Correlation pipeline is different than the rest as it requires many other parameters.** <br/>

## 1) Installation Steps 
1.1) Install *devtools* library on your R-studio environment.\
1.2) Run the the *Installation_Steps.R* file to install all the libraries.<br/>

## 2) Installation Testing and Execution 
2.1) Run the script *Implementation.R* using the command: *Rscript Implementation.R -infile -outdir* to check the default implementation.\
2.2) *-infile* refers to the location of the datafile that needs to be executed.<br/>
2.3) *-outdir* refers to the location of the output directory where the program output file will be generated.<br/>

## 3) Bulk Execution of Pipelines
3.1) Navigate to the pipeline folder and then to the algorithm you want to execute on an entire dataset. \
3.2) Open the python file starting with name 'Execute_'. <br/>
3.3) Change the dataset path and output path carefully. Also, beware of the name change happeing inside the loop. <br/>
3.4) Execute the python script after editing.

*Note:* The script folder contains the *.sh* scrpits for file execution. You can use that
if you want.

## 4) Execution of Consolidated_Analytics_Pipeline
4.1) The pipeline executed through the main execution file named 'Compute_Analytics.py'. <br/>
4.2) Set the parameters in the 'Compute_Analytics' file and execute it.<br/>
4.3) This file with run and generate the consolidated analytics of the pipeline. <br/>