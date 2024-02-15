#!/bin/bash
#proceso CD-HIT (Septiembre 2022)
#Alida Zarate

# Como correr qsub merge_cluster_agosto.sh /scratch/alida_uaem/cluster/Results5/procariontes
#1 Path principal /scratch/alida_uaem/cluster/Results5/procariontes


# Nombre del job
#$ -N Merge-Clust_host
#$ -j y
#$ -o /scratch/alida_uaem/salidas

#Interumpir por cualquier falla
set -e

# PATH para cdhit
source $HOME/.bashrc
module load programs/cdhit-4.8.1
cd $1

echo "Inicio Proceso merge:"


#Merge1
#clstr_rev.pl $1/nr80.clstr $1/nr60.clstr > $1/nr80-60.clstr 

#Merge2
clstr_rev.pl $1/nr80-60.clstr $1/nr30.clstr > $1/nr80-60-30.clstr

echo "Finalizo el merge"
