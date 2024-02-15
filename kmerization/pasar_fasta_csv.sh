
# Programa sh llama a perl 
#Programa pasar_fasta_csv.sh que llama a pasar_fasta_csv.pl

#$ -N fasta_to_csv
#$ -j y
#$ -o /scratch/alida_uaem/salidas

#qsub pasar_fasta_csv.sh

source $HOME/.bashrc
source /share/apps/Profiles/share-profile.sh

module load compilers/perl-5.28

perl  /scratch/alida_uaem/pasar_fasta_csv.pl
#perl  /scratch/alida_uaem/hola.pl

