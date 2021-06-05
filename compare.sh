#!/bin/sh

rm -rf log.txt
make dataclean
rm -rf *.ELL
./dw_exact_sxsx.exec
mv log.txt log.dw
./computecorr.exec 1 0 3 100 0.01 corr.dat > corr.lanc
#./allT.exec sz.dat > sz.lanc
./allT.exec energy.dat > energy.lanc
#./allT.exec energy2.dat > energy2.lanc

make dataclean
rm -rf *.ELL
./2dw_exact_sxsx.exec
mv log.txt log.2dw
./computecorr.exec 1 0 3 100 0.01 corr.dat > corr.exact
#./allT.exec sz.dat > sz.exact
./allT.exec energy.dat > energy.exact
#./allT.exec energy2.dat > energy2.exact

xmgrace -param corr.par corr.exact corr.lanc &
xmgrace -param beta.par energy.exact energy.lanc 
#xmgrace -param beta.par energy2.exact energy2.lanc 
#xmgrace -param beta.par sz.exact sz.lanc 



