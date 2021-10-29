# Fusion_Approach_Pipelines
 This repo contains different implementations of microbial interaction network creation.
Each folder contains the default implementation of a research paper. Each folder contains
an Implementation file, Installation_Steps file and Github file referring to the source Github directory.<br/>

**Note: Xiao_and_Correlation pipeline is different than the rest as it requires many other parameters.** <br/>

## 1) Installation Steps 
1.1) Install *devtools* library on your R-studio environment.\
1.2) Run the the *Installation_Steps.R* file to install all the libraries.<br/>

## 2) Installation Testing and Execution 
2.1) Run the script *Implementation.R* using the command: *Rscript Implementation.R -infile -outdir* to check the default implementation.\
2.2) *-infile* refers to the location of the datafile that needs to be executed.<br/>
2.3) *-outdir* refers to the location of the output directory where the program output file will be generated.<br/>

## 3) Bulk Execution
3.1) Navigate to the pipeline folder and then to the algorithm you want to execute on an entire dataset. \
3.2) Open the python file starting with name 'Execute_' <br/>
3.3) Change the dataset path and output path carefully. Also, beware of the name change happeing inside the loop <br\>
3.4) Execute the python script after editing

*Note:* The script folder contains the *.sh* scrpits for file execution. You can use that
if you want 
