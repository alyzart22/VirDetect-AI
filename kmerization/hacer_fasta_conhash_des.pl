
my $inputids = $ARGV[0];
my $inputdb = $ARGV[1];
my $outputfasta = $ARGV[2];


## =================================== Main ===================================



##Instrucción para abrir un archivo en Perl la variable $! se establece para cuando falla una llamada al sistema
open (READ1, $inputdb) or die "Cannot open $inputdb: $!.\n";
my $description1 = <READ1>;


## ========================== Delaración de variables =========================



my $countids=0; 
my $count_acc=0; 
my $line = '';
my $aux= '';
my $countmatches=0; 
my $accessionid = '';
my $infoaccession = '';

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

    $IDkey{$accessionid} = ">".$accessionid."|".$linesplit[2]."\n".$linesplit[4]."\n";

}

close (READ1);


my ($line2);



#Se leer el archivo CSV
open (READ2, "$inputids") or die "Cannot open $inputids: $!.\n";


##Se escrine el archivo de salida
open(OUT, ">$outputfasta") or die "Couldn't open file $outputfasta\n";


#Bucle para leer el segundo archivo con los IDs 
 while($line2 = <READ2>) {

    @auxlist_ = split(" ", $line2);
    $line2 = $auxlist_[0];

    while ($i_ <5){
        print $line2;
        $i_++;
    }


    $count_acc++;
	 
    if(defined($IDkey{$line2})){

	    print OUT $IDkey{$line2};

	    $countmatches++;

     }
 }
 
 print "Numero de IDs en DB: $countids\n";
 print "Numero de IDs a recuperar: $count_acc\n";
 print "Numero de IDs con hits : $countmatches \n";
 print "Ejecucion finalizada...\n";

close READ2;
close OUT;

