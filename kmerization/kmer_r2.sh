#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Parametros:
	# $1: Path del fasta a kmerizar /scratch/alida_uaem/cluster/Results5/eucariontes30/euk30.fasta
	# $2: Path lista de clases /scratch/alida_uaem/cluster/Results5/eucariontes30/clases_secuencias_euk30.csv                     
	# $3: Path donde se guardaran los archivo fragmentados /scratch/alida_uaem/cluster/Results5/eucariontes30/euk30_100_100_kmers.csv
	# $4: size del kmer.ejemplo 100
	# $5: size del salto.ejemplo 10
#ejecutar JOB 
	#qsub kmer_r2.sh /scratch/alida_uaem/cluster/Results5/eucariontes30/euk30.fasta /scratch/alida_uaem/cluster/Results5/eucariontes30/clases_secuencias_euk30.csv /scratch/alida_uaem/cluster/Results5/eucariontes30/euk30_100_100_kmers.csv 100 10 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#!/bin/bash
#$ -N kmerizandoR
#$ -j y
#$ -o /scratch/alida_uaem/salidas


source $HOME/.bashrc
source /share/apps/Profiles/share-profile.sh
#module load programs/R-3.5.1
module load programs/R-4.1.2

# Para asegurar que todo va bien
echo "Kmerizando el archivo " 
echo $1
echo "----------- Parametros ---------"
echo " 1) Fasta a kmerizar:"
echo $1
total_seq= grep -c ">" $1
echo "Total de secuencias:"
echo $total_seq
echo " 2) ubicacion lista de clases:"
echo $2
echo "3 ) archivo de salida:"
echo $3
echo " 4) Tamano de kmers:"
echo $4
echo " 5) Tamano de salto:"
echo $5
echo "-------------------------------"

Rscript /scratch/alida_uaem/kmer_r2.R $1 $2 $3 $4 $5

total_kmers=$(cat $3 | wc -l )
echo "Se generaron un total de kmers: "
echo total_kmers 
