#!/bin/bash
#proceso psi cdhit (Sep2022)
#Alida Zarate

# Como correr qsub psi_host_euk.sh /scratch/alida_uaem/cluster/Results5/eucariontes 
#1 Path principal /scratch/alida_uaem/cluster/Results5/eucariontes


# Nombre del job
#$ -N psi_euk
#$ -j y
#$ -o /scratch/alida_uaem/salidas
#$ -l lenta
#$ -l h_vmem=64G


#Interumpir por cualquier falla
set -e

# PATH para cdhit
source $HOME/.bashrc
module load programs/cdhit-4.8.1
cd $1

echo "Inicio Proceso:"


#Paso 3
echo "PASO 3 40----------------------------------------------------------------------------------------------#####"
#psi-cd-hit.pl -i $1/Results/nr60 -o $1/Results/nr40 -c 0.4
#-G 0 alineamiento global, -aS COVERTURA para asegurarse q el alineamiento al menos cubre el 70% de la secuencia, -g 1 es escoger el algoritmo rapido, -prog blastp es default
#-G 0 -aS 0.80 vamos a usar G1 q es global por lo que no es necesario usar esta linea
psi-cd-hit.pl -i nr60 -o nr30 -c 0.3  -G 0 -aS 0.80 -g 1 -para 16 -blp 4

echo "Finalizo"

