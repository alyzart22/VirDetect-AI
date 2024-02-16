<em> <h1 align="center"> DeepEukVirProt </h1></em>

This repository contains a Deep Learning model for identifying partial virus protein sequences in metagenomic data.
In this repository are available the necessary data and the environment to run the query application.

   <p align="left">
   <img src="https://img.shields.io/badge/STATUS-EN%20DESAROLLO-green">
   </p>


 [![Watchers](https://img.shields.io/github/watchers/alyzart22/DeepEukVirProt.svg)](https://github.com/alyzart22/DeepEukVirProt/watchers)
[![Stars](https://img.shields.io/github/stars/alyzart22/DeepEukVirProt.svg)](https://github.com/alyzart22/DeepEukVirProt/stargazers)
[![Activity](https://img.shields.io/github/commit-activity/m/alyzart22/DeepEukVirProt.svg)](https://github.com/DeepEukVirProt/DeepEukVirProt/commits)


<!-- Model Deep Learning architecture-->
# Architecture of Model Deep Learning
 ![model_image ](https://github.com/alyzart22/DeepEukVirProt/blob/main/img/modelo_300_980.jpg)


<!-- INSTALL API -->
# Install API consult


1. Clone the repository to local (or download manually all repository)
   ```sh
   git clone https://github.com/alyzart22/DeepEukVirProt.git
   
   ```

## If you have GPU Nvidia GTX or RTX _*Drivers Nvidia should be updated_

2. Create enviroment
   ```sh
   conda env create --file ./API_deepeukvirprot/enviroments/deepeukvirprot_gpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate deepeukvirprot_gpu 
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

## If you don´t have GPU
2. Create enviroment
   ```sh
   conda env create --file ./API_deepeukvirprot/enviroments/deepeukvirprot_cpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate deepeukvirprot_cpu 
   ```
## Execute API consult DeepEukVirProt
3. In this section you can try with you own metagenomics data
   In this line you can replace the fasta file unknown.fasta for your own fasta.
   Execute this line.
   ```sh
   python3 ./API_deepeukvirprot/api_deepeukvirprot.py ./API_deepeukvirprot/metagenomic_data/unknown/unknown.fasta ./API_deepeukvirprot/model.h5 ./API_deepeukvirprot/ref_api_300_20_980.csv ./API_deepeukvirprot/metagenomic_data/unknown/ 300 40 label_output_ 0.80 0.90 978 979 0 
   ```
## Output Api consult DeepEukVirProt
 4. The output are the following 6 images and 2 csv report with the predictions by kmers and by sequences.

 ![Output image ](https://github.com/alyzart22/DeepEukVirProt/blob/main/img/output_model.jpg)

<!-- REFERENCE -->
## Reference and citation
If you use DeepEukVirProt plese cite this paper:
[Doi paper here](https://github.com/alyzart22/DeepEukVirProt)


<!-- CONTACT -->
## Contact

Ali Zárate - alida.zarate@ibt.unam.mx

Project Link: [https://github.com/alyzart22/DeepEukVirProt](https://github.com/alyzart22/DeepEukVirProt)

## Autors

| [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Alida Zárate </sub>](https://github.com/alyzart22) | [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Lorena Diaz</sub>](https://github.com/alyzart22) | [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Blanca Taboada</sub>](https://github.com/alyzart22) |
| :---: | :---: | :---: |
