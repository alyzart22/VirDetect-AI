<em> <h1 align="center"> VirDetect-AI </h1></em>

This repository contains a Deep Learning model for identifying partial virus protein sequences in metagenomic data.
In this repository are available the necessary data and the environment to run the query application.

Download Extra suplementary data of VirDetect-AI
[https://zenodo.org/doi/10.5281/zenodo.13328820 ](https://zenodo.org/doi/10.5281/zenodo.13328820)

   <p align="left">
   <img src="https://img.shields.io/badge/STATUS-PASSED-green">
   </p>


 [![Watchers](https://img.shields.io/github/watchers/alyzart22/DeepEukVirProt.svg)](https://github.com/alyzart22/VirDetect-AI/watchers)
[![Stars](https://img.shields.io/github/stars/alyzart22/DeepEukVirProt.svg)](https://github.com/alyzart22/VirDetect-AI/stargazers)
[![Activity](https://img.shields.io/github/commit-activity/m/alyzart22/DeepEukVirProt.svg)](https://github.com/VirDetect-AI/VirDetect-AI/commits)


<!-- Model Deep Learning architecture-->
# Architecture of Model Deep Learning
 ![model_image ](https://github.com/alyzart22/VirDetect-AI/blob/main/img/modelo_300_980.jpg)


<!-- Api consult VirDetect-AI-->
# Api consult VirDetect-AI
 ![model_image ](https://github.com/alyzart22/VirDetect-AI/blob/main/img/Api_virdetect-ai.JPG)

<!-- Options-->
# There are two options to test the VirDetect-AI tool, through a google colab notebook or locally by installing a predefined environment
<!-- Execute notebook -->
## Option 1 - Execute Notebook
1.- Download the notebook Notebook_api_VirDetect-AI.ipynb located in the Notebook_VirDetect-AI folder in this repository
2.- Execute the notebook Notebook_api_VirDetect-AI.ipynb on Google colab (GPU)  or jupiter. Remember the format allow is only Fasta and the output is generated and saved in the outputs folder, which is a temporary folder in google drive [content], remember to download your results.

<!-- INSTALL API -->
## Option 2 -Install API consult


1. Clone the repository to local (or download manually all repository)
   ```sh
   git clone https://github.com/alyzart22/VirDetect-AI.git
   
   ```
2.- Download from this link the model and colocate inside the folder API_VirDetect-AI
   ```sh
https://drive.google.com/file/d/1jVLshzOz3bOPWuIbaoNSAV4yRl4JBAOD/view?usp=sharing
   ```

### If you have GPU Nvidia GTX or RTX _*Drivers Nvidia should be updated_

2. Create enviroment
   ```sh
   conda env create --file ./API_VirDetect-AI/enviroments/virdetect-ai_gpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate virdetect-ai_gpu 
   ```
   Execute this line in console 
   ```sh
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ 
   ```
   
   Execute this line to check that the gpu is working 
   ```sh
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Output expected example: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### If you don´t have GPU
2. Create enviroment
   ```sh
   conda env create --file ./VirDetect-AI/enviroments/virdetect-ai_cpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate virdetect-ai_cpu 
   ```
### Execute API consult VirDetect-AI
3. In this section you can try with you own metagenomics data
   In this line you can replace the fasta file unknown.fasta for your own fasta.
   Execute this line.
   ```sh
   python3 ./VirDetect-AI/api_virdetect-ai.py ./API_VirDetect-AI/metagenomic_data/unknown/unknown.fasta ./API_VirDetect-AI/model.h5 ./API_virdetect-ai/ref_api_300_20_980.csv ./API_deepeukvirprot/metagenomic_data/unknown/ 300 40 label_output_ 0.80 0.90 978 979 0 
   ```
### Output Api consult VirDetect-AI
 4. The output are the following 6 pie graphs and 3 files csv, report with the predictions by kmers, prediction by sequences and sequences unknown.

 ![Output image ](https://github.com/alyzart22/VirDetect-AI/blob/main/img/fig_s1.jpg)

<!-- REFERENCE -->
# Reference and citation
If you use VirDetect-AI plese cite this paper:
[Doi paper here](https://github.com/alyzart22/VirDetect-AI)


<!-- CONTACT -->
# Contact

Ali Zárate - alida.zarate@ibt.unam.mx

Project Link: [https://github.com/alyzart22/VirDetect-AI](https://github.com/alyzart22/VirDetect-AI)

# Authors


 [![width="40px"](https://github.com/alyzart22/VirDetect-AI/blob/main/img/use_icon.png)](https://www.researchgate.net/profile/Alida-Zarate)
 [![alt test](https://github.com/alyzart22/VirDetect-AI/blob/main/img/use_icon.png)](https://www.researchgate.net/profile/Blanca-Taboada)
 [![alt test](https://github.com/alyzart22/VirDetect-AI/blob/main/img/use_icon.png)](https://www.researchgate.net/profile/Lorena-Diaz-Gonzalez) 
 #### Alida Zárate |  Blanca Taboada  | Lorena Díaz  
