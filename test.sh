#!/bin/sh

make dataclean
rm -rf *.ELL
rm -rf log.txt
cp randomseed.in randomseed
#cp sectorstosample.in sectorstosample
./2dw_exact_sxsx.exec > out.2dw
mv log.txt log.2dw
mv corr.dat corr.2dw
mv Z.dat Z.2dw

make dataclean
rm -rf *.ELL
rm -rf log.txt
cp randomseed.in randomseed
#cp sectorstosample.in sectorstosample
./dw_exact_sxsx.exec > out.dw
mv log.txt log.dw
mv corr.dat corr.dw
mv Z.dat Z.dw
