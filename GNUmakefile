SHELL=/bin/bash

DATADIR= /scratch/sylju
HOMEDIR= /mn/felt/u20/sylju
ABELDIR= $(HOME)

include makefile


%.x64:	force
	cp Makefile.x64 Makefile.local
	echo `svnversion`
	make -f makefile $*.exec
	cp $*.exec $(DATADIR)/bin/$*.`svnversion`_x64.x

%.ox64:	force
	cp Makefile.x64 Makefile.local
	echo `svnversion`
	make -f makefile $*.o
#	mv $*.exec $(DATADIR)/bin/$*.`svnversion`_x64.x

%.osx:	force
	cp Makefile.osx Makefile.local
	echo `svnversion`
	make -f makefile $*.exec
#	mv $*.exec $(DATADIR)/bin/$*.`svnversion`_x64.x


%.386: 	force
	cp Makefile.386 Makefile.local
	echo `svnversion`
	make -f makefile $*.exec
	mv $*.exec $(DATADIR)/bin/$*.`svnversion`_386.x

%.abel:	force
	cp Makefile.abel Makefile.local
	echo `svnversion`
	make -f makefile $*.exec
	cp $*.exec $(ABELDIR)/bin/$*.`svnversion`_abel.x

#%:	force
#	make -f makefile $*


allx64 : computediag_oneT.x64 computediag_allT.x64 computecorr_oneT.x64 computecorr_allT.x64 dw_lanczos_as.x64 dw_lanczos_proj_as.x64 dw_lanczos_sxsx_as.x64 dw_lanczos_sysy_as.x64 dw_lanczos_szsz_as.x64 dw_lanczos_sxsx_as_oneT.x64 dw_lanczos_sysy_as_oneT.x64 dw_lanczos_szsz_as_oneT.x64 dw_lanczos.x64 dw_lanczos_proj.x64 dw_lanczos_sxsx.x64 dw_lanczos_sysy.x64 dw_lanczos_szsz.x64 dw_lanczos_sxsx_oneT.x64 dw_lanczos_sysy_oneT.x64 dw_lanczos_szsz_oneT.x64 dw_exact.x64 dw_exact_oneT.x64 dw_exact_proj.x64 dw_exact_sxsx.x64 dw_exact_sysy.x64 dw_exact_szsz.x64 dw_exact_sxsx_oneT.x64 dw_exact_sysy_oneT.x64 dw_exact_szsz_oneT.x64 2dw_exact_sxsx.x64 

allabel : computediag_oneT.abel computediag_allT.abel computecorr_oneT.abel computecorr_allT.abel dw_lanczos_as.abel dw_lanczos_proj_as.abel dw_lanczos_sxsx_as.abel dw_lanczos_sysy_as.abel dw_lanczos_szsz_as.abel dw_lanczos_sxsx_as_oneT.abel dw_lanczos_sysy_as_oneT.abel dw_lanczos_szsz_as_oneT.abel dw_lanczos.abel dw_lanczos_proj.abel dw_lanczos_sxsx.abel dw_lanczos_sysy.abel dw_lanczos_szsz.abel dw_lanczos_sxsx_oneT.abel dw_lanczos_sysy_oneT.abel dw_lanczos_szsz_oneT.abel dw_exact.abel dw_exact_oneT.abel dw_exact_proj.abel dw_exact_sxsx.abel dw_exact_sysy.abel dw_exact_szsz.abel dw_exact_sxsx_oneT.abel dw_exact_sysy_oneT.abel dw_exact_szsz_oneT.abel 2dw_exact_sxsx.abel dw_lanczos_sxsx_oneT_cuda.abel dw_lanczos_sysy_oneT_cuda.abel dw_lanczos_szsz_oneT_cuda.abel dw_lanczos_sxsx_as_oneT_cuda.abel dw_lanczos_sysy_as_oneT_cuda.abel dw_lanczos_szsz_as_oneT_cuda.abel dw_lanczos_cuda.abel dw_lanczos_proj_cuda.abel dw_lanczos_sxsx_cuda.abel dw_lanczos_sysy_cuda.abel dw_lanczos_szsz_cuda.abel dw_lanczos_as_cuda.abel dw_lanczos_proj_as_cuda.abel dw_lanczos_sxsx_as_cuda.abel dw_lanczos_sysy_as_cuda.abel dw_lanczos_szsz_as_cuda.abel 

testabel: dw_lanczos_sxsx.abel dw_lanczos_sxsx_cuda.abel 

testx64: dw_lanczos_sxsx.x64 dw_lanczos_sxsx_as.x64 

exactx64: 2dw_exact_sxsx.x64 dw_exact_sxsx.x64
exactabel: 2dw_exact_sxsx.abel dw_exact_sxsx.abel

tests: doubletest.abel complextest.abel doublerandomtest.abel complexrandomtest.abel

force : ;





