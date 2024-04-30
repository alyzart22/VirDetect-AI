
library(Biostrings) 
library(base)

#----------- Funciones--------------------------
find_kmer_salto<-function(string,k,step){
  n<-round((nchar(string)-k+1)/step)
  #print(n)
  
  kmers<-substring(string,seq(1,nchar(string)-k+1,step),seq(k,nchar(string),step))
  return (kmers)
}
#--------------------------------------------

args <- commandArgs(TRUE)
urlFa=args[1]
path_id_clase=args[2]
pathDestino=args[3]
k_mer=as.numeric(args[4])
v_step=as.numeric(args[5])

#imprimir la clase que le toca
id_clase <- read.csv(path_id_clase, header = FALSE)
#id_clase[3,]
#Se lee el archivo FASTA de entrada
arc<-readBStringSet(urlFa)
#Definir contador
cont=1
contador_clases=1
#Cálculo de la secuencia en el archivo FASTA
maximo=length(arc)
x<-"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


while(cont<=maximo){
  
  fastaStr=toString(arc[cont,])
  faSt=gsub(",","",fastaStr)
  faSt=gsub(" ","",faSt)
  #calcula la longitud de la secuencia
  longfaSt=nchar(faSt)
  #print(longfaSt)
  if (longfaSt < k_mer){
    #kmers=faSt
    falta=k_mer-longfaSt
    #print(falta)
    cadena_extra= substr(x, start = 1, stop = falta) 
    kmers = paste(faSt,cadena_extra,sep="")
  }else{
    #llama a la funcion de kmer con salto
    kmers=find_kmer_salto(faSt,k_mer,v_step)
  }
  #obtiene el identificador Genebank
  pathlistIDG=unlist(strsplit(names(arc[cont,]), " "))
  idGB=pathlistIDG[[1]]
  dfKmer=data.frame(id_clase[contador_clases,], kmers)
  #direcccion donde se guardara
  urlKmer=paste(pathDestino)
  write.table(dfKmer,urlKmer,col.names = FALSE,row.names = FALSE,append=TRUE, sep=',')
  
  cont=cont+1
  contador_clases=contador_clases+1
  
}
