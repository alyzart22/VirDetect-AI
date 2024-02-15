#!/bin/bash
#proceso CD-HIT (Septiembre 2022) hacer grupos pero de los fasta de los host ya definidos, si son euk o pro
#Alida Zarate

# Como correr qsub clusterizacion_host.sh /scratch/alida_uaem/cluster/Results5/eucariontes eucariontes.fasta
#1 Path principal /scratch/alida_uaem/cluster/Results5/eucariontes


# Nombre del job
#$ -N Cluster_by_host
#$ -pe thread 4
#$ -j y
#$ -o /scratch/alida_uaem/salidas

#Interumpir por cualquier falla
set -e

# PATH para cdhit
source $HOME/.bashrc
module load programs/cdhit-4.8.1
cd $1

echo "Inicio Proceso:"

#total_seqs=$(grep -c '>' /scratch/$USER/cluster/eucariontes/eucariontes.fasta) 
total_seqs=$(grep -c '>' $1/$2 )
echo "Total de secuencias iniciales de $2 son $total_seqs"


#Paso 1 cluster 80
echo "PASO 1 80----------------------------------------------------------------------------------------------#####"
cd-hit -i $1/$2 -o $1/nr80 -c 0.8 -n 5 -M 16000 -T 4


#Paso 2 cambio el parametro n por que bajo el umbral
echo "PASO 2 60----------------------------------------------------------------------------------------------#####"
cd-hit -i $1/nr80 -o $1/nr60 -c 0.6 -n 4 -d 0 -M 16000 -T 4

