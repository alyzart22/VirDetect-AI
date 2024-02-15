#!/bin/bash
# Programa sh llama a perl 
#Programa hacer_fasta_conhash.sh que llama a hacer_fasta_conhash.pl
# correr qsub hacer_fasta_conhash.sh /scratch/alida_uaem/cluster/Results5/eucariontes30/acc_euc30_part2.txt /scratch/alida_uaem/cluster/Results/all_dup_julio.fasta.csv /scratch/alida_uaem/cluster/Results5/eucariontes30/parte_2_euk30.fasta
#3 parametros de entrada


#$ -N hacer_fasta_hash
#$ -j y
#$ -o /scratch/alida_uaem/salidas

#qsub hacer_fasta_conhash.sh

source $HOME/.bashrc
source /share/apps/Profiles/share-profile.sh

module load compilers/perl-5.28

#perl  /scratch/alida_uaem/hacer_fasta_conhash.pl
perl  /scratch/alida_uaem/hacer_fasta_conhash_des.pl $1 $2 $3
