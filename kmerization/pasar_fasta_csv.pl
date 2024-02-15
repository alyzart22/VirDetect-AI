
my $inputdata = "/scratch/alida_uaem/cluster/Results/all_dup_julio.fasta";
my $output = "/scratch/alida_uaem/cluster/Results/all_dup_julio_2.fasta.csv";


## ============================ Declaración de variables ======================

########################################################################
my $cutoff="95%"; 		#Especificar el % del cutoff de kmers analizados
########################################################################
my $id=""; 	 	#Almacena el IDs de la secuencia
my $nokmer=0;		#Almacena el nokmer
my $sequence=""; 	#Almacena la secuencia de nt
my $line=""; 		#Línea para ir leyendo todo el archivo
my $expresion=">"; 	#Expresión para identificar los ID del archivo FASTA
my @split; 		#Lista para fragmentar por signo > y eliminarlo del ID
my $count=0; 		#Número de secuencias escritas en el archivo de salida
my @auxvacios;
my @auxcomas;


my ($description, $family, $host);
## =================================== Main ===================================
##Instrucción para abrir un archivo en Perl la variable $! se establece para cuando falla una llamada al sistema
open (READ1, $inputdata) or die "Cannot open $inputdata: $!.\n";

##Se escribe el archivo de salida
open(OUT, ">$output") or die "Couldn't open file $output\n";
print OUT "IDaccesion,description,family,host,sequence\n";

while($line = <READ1>){
	#Elimina retorno de carro de la primer línea leída
	chomp $line;
	#Si la línea comienza con signo >:
	if($line =~ /^\Q$expresion\E/){
		#Incrementa contador de secuencias analizadas
		$count++;
		#Split por signo > en ID de FASTA para eliminarlo
		@split = split(">",$line);
		#Se almacena ID de la secuencia
		$id= $split[1];
		#Extracción del nokmer desde el id de la secuencia del FASTA
		@split = split(/\|/, $id);
		#Extracción del nokmer

		#hacer split de espacios vacios para que quite el espacio generado despues del accesions
		$id = $split[0];
		@auxvacios = split(" ", $id);
    	$id = $auxvacios[0];


		$description =  $split[1];
		@auxcomas= split(",", $description);
		$description = $auxcomas[0];

		$family =  $split[2];
		$host =  $split[3];
		#Extracción de la siguiente línea correspondiente a la secuencia de nt
		$sequence = <READ1>;
		#Elimina retorno de carro de la línea extraída
		chomp $sequence;
		print OUT $id.",".$description.",".$family.",".$host.",".$sequence."\n";
		
	}
}
print "Total de secuencias escritas: $count\n";
print "Ejecucion finalizada...\n";

