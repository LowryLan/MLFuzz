rm -rf out
rm -rf weight_info
rm -rf weight_info_r

./afl-fuzz -i in3 -o out ./xmllint @@
