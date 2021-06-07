DATADIR= /scratch/sylju
#HOMEDIR= /mn/felt/u1/sylju
HOMEDIR= $(HOME)
ABELDIR= $(HOME)
SAGADIR= $(HOME)
OSXDIR = $(HOME)

%.x64 : force
	cp Makefile.x64 Makefile.local
	@echo `git describe`
	make -f makefile $*.exec

%.osx : force
	cp Makefile.osx Makefile.local
	@echo `git describe`
	make -f makefile $*.exec


%.saga : force
	cp Makefile.saga Makefile.local
	@echo `git describe`
	make -f makefile $*.exec

%.saga : force
	cp Makefile.saga Makefile.local
	@echo `git describe`
	make -f makefile $*.exec

# The envirmoment variable MYMAKEFILE contains the name of the appropriate Makefile.local
%    :  force
	cp $(MYMAKEFILE) Makefile.local 
	make -f makefile $*




force : ;


allx64 : computediag_oneT.x64 computediag_allT.x64 computecorr_oneT.x64 computecorr_allT.x64 dw_lanczos_as.x64 dw_lanczos_proj_as.x64 dw_lanczos_sxsx_as.x64 dw_lanczos_sysy_as.x64 dw_lanczos_szsz_as.x64 dw_lanczos_sxsx_as_oneT.x64 dw_lanczos_sysy_as_oneT.x64 dw_lanczos_szsz_as_oneT.x64 dw_lanczos.x64 dw_lanczos_proj.x64 dw_lanczos_sxsx.x64 dw_lanczos_sysy.x64 dw_lanczos_szsz.x64 dw_lanczos_sxsx_oneT.x64 dw_lanczos_sysy_oneT.x64 dw_lanczos_szsz_oneT.x64 dw_exact.x64 dw_exact_oneT.x64 dw_exact_proj.x64 dw_exact_sxsx.x64 dw_exact_sysy.x64 dw_exact_szsz.x64 dw_exact_sxsx_oneT.x64 dw_exact_sysy_oneT.x64 dw_exact_szsz_oneT.x64 2dw_exact_sxsx.x64 

allsaga : computediag_oneT.saga computediag_allT.saga computecorr_oneT.saga computecorr_allT.saga dw_lanczos_as.saga dw_lanczos_proj_as.saga dw_lanczos_sxsx_as.saga dw_lanczos_sysy_as.saga dw_lanczos_szsz_as.saga dw_lanczos_sxsx_as_oneT.saga dw_lanczos_sysy_as_oneT.saga dw_lanczos_szsz_as_oneT.saga dw_lanczos.saga dw_lanczos_proj.saga dw_lanczos_sxsx.saga dw_lanczos_sysy.saga dw_lanczos_szsz.saga dw_lanczos_sxsx_oneT.saga dw_lanczos_sysy_oneT.saga dw_lanczos_szsz_oneT.saga dw_exact.saga dw_exact_oneT.saga dw_exact_proj.saga dw_exact_sxsx.saga dw_exact_sysy.saga dw_exact_szsz.saga dw_exact_sxsx_oneT.saga dw_exact_sysy_oneT.saga dw_exact_szsz_oneT.saga 2dw_exact_sxsx.saga dw_lanczos_sxsx_oneT_cuda.saga dw_lanczos_sysy_oneT_cuda.saga dw_lanczos_szsz_oneT_cuda.saga dw_lanczos_sxsx_as_oneT_cuda.saga dw_lanczos_sysy_as_oneT_cuda.saga dw_lanczos_szsz_as_oneT_cuda.saga dw_lanczos_cuda.saga dw_lanczos_proj_cuda.saga dw_lanczos_sxsx_cuda.saga dw_lanczos_sysy_cuda.saga dw_lanczos_szsz_cuda.saga dw_lanczos_as_cuda.saga dw_lanczos_proj_as_cuda.saga dw_lanczos_sxsx_as_cuda.saga dw_lanczos_sysy_as_cuda.saga dw_lanczos_szsz_as_cuda.saga 

lanczosx64: dw_lanczos_sxsx.x64 dw_lanczos_sxsx_as.x64 
lanczossaga: dw_lanczos_sxsx.saga dw_lanczos_sxsx_cuda.saga 

exactx64: 2dw_exact_sxsx.x64 dw_exact_sxsx.x64
exactsaga: 2dw_exact_sxsx.saga dw_exact_sxsx.saga

testsaga: doubletest.saga complextest.saga doublerandomtest.saga complexrandomtest.saga
testx64: doubletest.x64 complextest.x64 doublerandomtest.x64 complexrandomtest.x64


