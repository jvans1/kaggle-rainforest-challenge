
BEGIN{
  FS=","
  }{
  if(NR==1){
    print $start , $(start+1), $(start+33), $(start+34)
    }else{
    print substr($start,0,7) "              " substr($(start+1),0,7) "                "  substr($(start+33),0,7)"                 "  substr($(start+34),0,7)
  }
}END{}
