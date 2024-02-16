<em> <h1 align="center"> DeepEukVirProt </h1></em>

This repository contains a Deep Learning model for identifying partial virus protein sequences in metagenomic data.
In this repository are available the necessary data and the environment to run the query application.

   <p align="left">
   <img src="https://img.shields.io/badge/STATUS-EN%20DESAROLLO-green">
   </p>

<!-- INSTALL API -->
# Install API consult


1. Clone the repository to local (or download manually all repository)
   ```sh
   git clone https://github.com/alyzart22/DeepEukVirProt.git
   
   ```

## If you have GPU 
2. Create enviroment
   ```sh
   conda env create --file ./API_deepeukvirprot/enviroments/deepeukvirprot_gpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate deepeukvirprot_gpu 
   ```
   Copy this line in console ()
   ```sh
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ 
   ```
   Execute this line to check that the gpu is working 
   ```sh
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

## If you don´t have GPU
2. Create enviroment
   ```sh
   conda env create --file ./API_deepeukvirprot/enviroments/deepeukvirprot_cpu.yml 
   ```
   Activate you enviroment
   ```sh
   conda activate deepeukvirprot_cpu 
   ```

Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
3. 

<!-- REFERENCE -->
## Reference and citation
If you use DeepEukVirProt plese cite this paper:
[https://github.com/alyzart22/DeepEukVirProt](https://github.com/alyzart22/DeepEukVirProt)


<!-- CONTACT -->
## Contact

Ali Zárate - alida.zarate@ibt.unam.mx

Project Link: [https://github.com/alyzart22/DeepEukVirProt](https://github.com/alyzart22/DeepEukVirProt)

## Autors

| [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Alida Zárate </sub>](https://github.com/alyzart22) | [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Lorena Diaz</sub>](https://github.com/alyzart22) | [<img src="https://png.pngtree.com/png-clipart/20191122/original/pngtree-user-icon-isolated-on-abstract-background-png-image_5192004.jpg" width=115><br><sub>Blanca Taboada</sub>](https://github.com/alyzart22) |
| :---: | :---: | :---: |
