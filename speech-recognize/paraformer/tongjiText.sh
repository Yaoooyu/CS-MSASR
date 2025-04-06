#!/bin/bash
 
rm -f textTotal.txt
rm -f tokenTotal.txt

for i in {1..108}
do
    echo "--$dir"
    dir="test/debug""$i""/1best_recog"
    textFile="$dir""/text"
    tokenFile="$dir""/token"

    cat $textFile >>textTotal.txt
    cat $tokenFile >>tokenTotal.txt
done
