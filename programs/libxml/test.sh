export AFL=/home/lowry/Documents/myFuzz/MLFuzz/afl-lowry

rm -rf out
$AFL/afl-fuzz -i in2 -o out ./xmllint @@