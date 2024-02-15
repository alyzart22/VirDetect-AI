#author: Edna Cruz Fl version modificada 4 sep 2022

## =============  Variables para rutas de los archivos input/output ===========
##Rutas para secuencias DR
#my $inputids = "C:\\Users\\EdnaCruzFlores\\Downloads\\db_ali\\accesions_euk.csv";
#my $inputdb = "C:\\Users\\EdnaCruzFlores\\Downloads\\db_ali\\all_dup_julio.fasta.csv";
#my $outputfasta = "C:\\Users\\EdnaCruzFlores\\Downloads\\db_ali\\eucariontes.fasta";

#my $inputids = "/scratch/alida_uaem/cluster/Results5/eucariontes/accesions_euk.csv";
#my $inputdb = "/scratch/alida_uaem/cluster/Results/all_dup_julio_2.fasta.csv";
#my $outputfasta = "/scratch/alida_uaem/cluster/Results5/eucariontes/eucariontes.fasta";

my $inputids = $ARGV[0];
my $inputdb = $ARGV[1];
my $outputfasta = $ARGV[2];


## =================================== Main ===================================

	    #########  PROCESAR IDs de Estructuras CRISPR y generar tabla hash  ############

##Instrucción para abrir un archivo en Perl la variable $! se establece para cuando falla una llamada al sistema
open (READ1, $inputdb) or die "Cannot open $inputdb: $!.\n";
my $description1 = <READ1>;


## ========================== Delaración de variables =========================

#string para TaxID

my $countids=0; #Contador de IDs de virus
my $countphages=0; #Contador de IDs de fagos
my $line = '';
my $aux= '';
my $countmatches=0; #Contador de los hits que se encontraron en común en phages y virueses dbs
my $accessionid = '';
my $infoaccession = '';
#lista para split de las líneas del archivo de IDs de virus
my @linesplit;
my @auxlist_;
#tabla hash
my %IDkey = ();
my $i=0;
my $i_=0;
#Lectura del archivo por linea
while ($line = <READ1>){
    #Se elimina el retorno de carro de la linea
    chomp $line;
    #Incrementa contador de IDs de virus
    $countids++;
    #Se hace split por ,
    @linesplit = split(",", $line);
    #Se extrae la primera posición de la lista obtenida tras el split
    $accessionid = $linesplit[0];
    #@auxlist = split(" ", $accessionid);
    #$accessionid = $auxlist[0];
    
    #print $accessionid;

   # while ($i <5){ print $accessionid; $i++;}

    #Se aplica un segundo split ahora por . para quitar la versión del ID
    #@auxlist = split("\\.",$aux);
    #Almacenamiento del ID sin puntos de la versión
    #print ">".$accessionid."|".$linesplit[1]."|".$linesplit[2]."|".$linesplit[3]."\n".$linesplit[4]."\n";
    #El ID es utilizado como key y como dato se vuelve a colocar la línea con toda la info del virus 
    #$IDkey{$accessionid} = ">".$accessionid."\n".$linesplit[4]."\n";
    $IDkey{$accessionid} = ">".$accessionid."|".$linesplit[2]."\n".$linesplit[4]."\n";
    #$IDkey{$accessionid} = ">".$accessionid."\n";
}

close (READ1);


	    #########  PROCESAR ARCHIVO CSV CON LOS ID DE LOS HITS DE PHAGE-IN-NT-50BP  ############

my ($line2);
#my @linea;


#Se leer el archivo CSV
open (READ2, "$inputids") or die "Cannot open $inputids: $!.\n";


##Se escrine el archivo de salida
open(OUT, ">$outputfasta") or die "Couldn't open file $outputfasta\n";


#Bucle para leer el segundo archivo con los IDs de phages
 while($line2 = <READ2>) {
    #Se elimina retorno de carro
    #chomp $line2;
    @auxlist_ = split(" ", $line2);
    $line2 = $auxlist_[0];

    while ($i_ <5){
        print $line2;
        $i_++;
    }

    #Incrementa contador de IDs fagos
    $countphages++;
    #print $line2;
    #print $idphage."\n";
    #Si lo almacenado en la posición 0 de la lista en la fila leída, coincide con una llave en la tabla hash %IDkey 	 
    if(defined($IDkey{$line2})){
	    #Se escribe la Info del reporte para esa estructura CRISPR de la matriz a omitir
	    print OUT $IDkey{$line2};
	    #incrementa contador de estructuras omitidas
	    $countmatches++;

     }
 }
 
 print "Numero de IDs en DB: $countids\n";
 print "Numero de IDs a recuperar: $countphages\n";
 print "Numero de IDs con hits : $countmatches \n";
 print "Ejecucion finalizada...\n";

close READ2;
close OUT;

