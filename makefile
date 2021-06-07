include Makefile.local

#for debugging purposes, add the switch -DDEBUG
#for more detailed debug information -DDEBUG2 
#PRECISION is just a variable for defining how accurate a double is printed on screen (for debugging)

SYSTEM=GNUmakefile makefile Makefile.x64 Makefile.saga



#GLOBALFLAGS = -DDEBUG -DDEBUG2 -DPRECISION=20 -DELLMATRIX
#GLOBALFLAGS = -DDEBUG -DELLMATRIX
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASBINARIES
#GLOBALFLAGS = -DELLMATRIX -DPRECISION=10 -DSPINPARITY_CONSERVED -DSTOREDATAASBINARIES -DSTOREDATAASFLOAT
#GLOBALFLAGS = -DELLMATRIX -DDEBUG -DDEBUG2 -DPRECISION=16 -DSPINPARITY_CONSERVED -DSTOREDATAASFLOAT -DDOUBLEDOUBLE
#GLOBALFLAGS = -DELLMATRIX -DDEBUG -DDEBUG2 -DPRECISION=16 -DSPINPARITY_CONSERVED -DSTOREDATAASFLOAT -DQUADDOUBLE
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASFLOAT -DSTOREDATAASBINARIES -DDEBUGPRECISION=40 -DDOUBLEDOUBLE -DDEBUG -DDEBUG2
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASFLOAT -DDEBUGPRECISION=16 -DDOUBLEDOUBLE -DDEBUG -DDEBUG2
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASFLOAT -DSTOREDATAASBINARIES -DDEBUGPRECISION=16 -DQUADDOUBLE
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASFLOAT -DDEBUGPRECISION=40 -DDOUBLEDOUBLE -DSTOREDATAASBINARIES
#GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASDOUBLE -DDEBUGPRECISION=16 -DDOUBLEDOUBLE -DSTOREDATAASBINARIES


GLOBALFLAGS = -DELLMATRIX -DSTOREDATAASDOUBLE -DDEBUGPRECISION=16 -DDOUBLEDOUBLE

%.run  : %.exec
	./$<

%.exec : %.o 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo $*.`git describe`$(EXTENSION)
	mv $*.exec $(BINDIR)/$*.`git describe`$(EXTENSION)


#%.exec : %.o 
#	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

myprog.exec : $(MYPROGOBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

myprogf.exec : $(MYPROGFOBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

mycuda.exec : $(MYCUDAOBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)


dottest.exec : dottest.o cuda_global.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

testortho.exec : testortho.o lowlevelroutines.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

gqdtest.exec : gqdtest.o cuda_global.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

doublerandomtest.exec : doublerandomtest.o cuda_precision.o cuda_global.o cuda_precision_kernels.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

doubletest.exec : doubletest.o cuda_precision.o cuda_global.o cuda_precision_kernels.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

complexrandomtest.exec : cuda_global.o cuda_precision.o cuda_precision_kernels.o complexrandomtest.o 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

normcheck.exec : norm.o cuda_precision.o cuda_global.o cuda_precision_kernels.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

complextest.exec : complextest.o cuda_precision.o cuda_global.o cuda_precision_kernels.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

Zaxpytest.exec : Zaxpytest.o cuda_precision.o cuda_global.o cuda_precision_kernels.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

computediag_oneT.exec : computediag_oneT.o observablesspec.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

computediag_allT.exec : computediag_allT.o 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

computecorr_oneT.exec : computecorr_oneT.o observablesspec.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

computecorr_allT.exec : computecorr_allT.o
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)



#####
TESTAS_OBJECTS = dwalls_as.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_as.exec : $(TESTAS_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTASP_OBJECTS = dwalls_proj_as.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_proj_as.exec : $(TESTASP_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTXAS_OBJECTS = dwalls_corr_sxsx_as.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sxsx_as.exec : $(TESTXAS_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTYAS_OBJECTS = dwalls_corr_sysy_as.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sysy_as.exec : $(TESTYAS_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTZAS_OBJECTS = dwalls_corr_szsz_as.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_szsz_as.exec : $(TESTZAS_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)


TEST_OBJECTS = dwalls.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos.exec : $(TEST_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTP_OBJECTS = dwalls_proj.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_proj.exec : $(TESTP_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTX_OBJECTS = dwalls_corr_sxsx.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sxsx.exec : $(TESTX_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTY_OBJECTS = dwalls_corr_sysy.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sysy.exec : $(TESTY_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTZ_OBJECTS = dwalls_corr_szsz.o  sparsematrix.o observables_std_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_szsz.exec : $(TESTZ_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)


TESTXAS1_OBJECTS = dwalls_corr_sxsx_as.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sxsx_as_oneT.exec : $(TESTXAS1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTYAS1_OBJECTS = dwalls_corr_sysy_as.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sysy_as_oneT.exec : $(TESTYAS1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTZAS1_OBJECTS = dwalls_corr_szsz_as.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_szsz_as_oneT.exec : $(TESTZAS1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)


TESTX1_OBJECTS = dwalls_corr_sxsx.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sxsx_oneT.exec : $(TESTX1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTY1_OBJECTS = dwalls_corr_sysy.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_sysy_oneT.exec : $(TESTY1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTZ1_OBJECTS = dwalls_corr_szsz.o  sparsematrix.o observables_std_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o lanczos_std.o observablesspec.o

dw_lanczos_szsz_oneT.exec : $(TESTZ1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)




TESTF_OBJECTS = dwalls_full.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact.exec : $(TESTF_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTF1_OBJECTS = dwalls_full.o  sparsematrix.o observables_std_full_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_oneT.exec : $(TESTF1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTFP_OBJECTS = dwalls_full_proj.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_proj.exec : $(TESTFP_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTFX_OBJECTS = dwalls_full_corr_sxsx.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_sxsx.exec : $(TESTFX_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo dw_exact_sxsx.`git describe`$(EXTENSION)
	mv dw_exact_sxsx.exec $(BINDIR)/dw_exact_sxsx.`git describe`$(EXTENSION)

TESTFY_OBJECTS = dwalls_full_corr_sysy.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_sysy.exec : $(TESTFY_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo dw_exact_sysy.`git describe`$(EXTENSION)
	mv dw_exact_sysy.exec $(BINDIR)/dw_exact_sysy.`git describe`$(EXTENSION)

TESTFZ_OBJECTS = dwalls_full_corr_szsz.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_szsz.exec : $(TESTFZ_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)
	@echo dw_exact_szsz.`git describe`$(EXTENSION)
	mv dw_exact_szsz.exec $(BINDIR)/dw_exact_szsz.`git describe`$(EXTENSION)

TESTFX1_OBJECTS = dwalls_full_corr_sxsx.o  sparsematrix.o observables_std_full_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o  observablesspec.o

dw_exact_sxsx_oneT.exec : $(TESTFX1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTFY1_OBJECTS = dwalls_full_corr_sysy.o  sparsematrix.o observables_std_full_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_sysy_oneT.exec : $(TESTFY1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

TESTFZ1_OBJECTS = dwalls_full_corr_szsz.o  sparsematrix.o observables_std_full_corr_oneT.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

dw_exact_szsz_oneT.exec : $(TESTFZ1_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

#####
2FX_OBJECTS = 2dw_full_corr_sxsx.o  sparsematrix.o observables_std_full_corr.o inttostring.o lowlevelroutines.o operator.o banddiag.o fulldiagonalization.o observablesspec.o

2dw_exact_sxsx.exec : $(2FX_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS)

#####


TEST_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_cuda.exec : $(TEST_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTP_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_proj.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_proj_cuda.exec : $(TEST_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTX_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_sxsx.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_sxsx_cuda.exec : $(TESTX_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTY_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_sysy.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_sysy_cuda.exec : $(TESTY_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTZ_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_szsz.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_szsz_cuda.exec : $(TESTZ_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTAS_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_as_cuda.exec : $(TESTAS_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTASP_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_proj_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_proj_as_cuda.exec : $(TESTASP_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTXAS_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_sxsx_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_sxsx_as_cuda.exec : $(TESTXAS_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTYAS_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_sysy_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_sysy_as_cuda.exec : $(TESTYAS_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTZAS_CUDA_OBJECTS = observables_cuda_corr.o lanczos_cuda.o dwalls_corr_szsz_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o

dw_lanczos_szsz_as_cuda.exec : $(TESTZAS_CUDA_OBJECTS) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)


TESTX_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_sxsx.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_sxsx_oneT_cuda.exec : $(TESTX_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTY_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_sysy.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_sysy_oneT_cuda.exec : $(TESTY_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTZ_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_szsz.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_szsz_oneT_cuda.exec : $(TESTZ_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)


TESTXAS_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_sxsx_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_sxsx_as_oneT_cuda.exec : $(TESTXAS_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTYAS_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_sysy_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_sysy_as_oneT_cuda.exec : $(TESTYAS_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)

TESTZAS_CUDA_OBJECTS1 = lanczos_cuda.o dwalls_corr_szsz_as.o  sparsematrix.o inttostring.o lowlevelroutines.o operator.o banddiag.o observablesspec.o cuda_precision.o cuda_precision_kernels.o cuda_global.o cuda_routines.o cuda_kernels.o observables_cuda_corr_oneT.o

dw_lanczos_szsz_as_oneT_cuda.exec : $(TESTZAS_CUDA_OBJECTS1) 
	$(CCC) $(LINKOPTS) -o $@ $^ $(LIBDIR) $(LIBS) $(CUDALIBDIR) $(CUDALIBS)



#####




################################## Rules for making object files #####################################

#####

DWALLSOURCES= exact.C global.h timer.h samplingcounter.h sparsematrix.h lanczos.h observables.h dwmodel.h inttostring.h statespace.h bitmanip.h twodomainstatesmanip.h xyzhamiltonian1d.h sparsematrix.h operator.h spluss.h sx.h sy.h sz.h projectionoperator.h hash.h RunParameter.h observablesspec.h

dwalls_as.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DUSEBASISSTATESASINITFORSMALLSECTORS -DSMALLSECTORLIMIT=1000 -c -o $@ $<

dwalls_proj_as.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCALCULATE_PROJECTIONOPS -DUSEBASISSTATESASINITFORSMALLSECTORS -DSMALLSECTORLIMIT=1000 -c -o $@ $<

dwalls_corr_sxsx_as.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSXSX -DUSEBASISSTATESASINITFORSMALLSECTORS -DSMALLSECTORLIMIT=1000 -c -o $@ $<

dwalls_corr_sysy_as.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSYSY -DUSEBASISSTATESASINITFORSMALLSECTORS -DSMALLSECTORLIMIT=1000 -c -o $@ $<

dwalls_corr_szsz_as.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSZSZ -DUSEBASISSTATESASINITFORSMALLSECTORS -DSMALLSECTORLIMIT=1000 -c -o $@ $<


dwalls.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -c -o $@ $<

dwalls_proj.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCALCULATE_PROJECTIONOPS -c -o $@ $<

dwalls_corr_sxsx.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSXSX -c -o $@ $<

dwalls_corr_sysy.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSYSY -c -o $@ $<

dwalls_corr_szsz.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSZSZ -c -o $@ $<


dwalls_full.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DFULLDIAGONALIZATION -c -o $@ $<

dwalls_full_proj.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DFULLDIAGONALIZATION -DCALCULATE_PROJECTIONOPS -c -o $@ $<

dwalls_full_corr_sxsx.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSXSX -DFULLDIAGONALIZATION -c -o $@ $<

dwalls_full_corr_sysy.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSYSY -DFULLDIAGONALIZATION -c -o $@ $<

dwalls_full_corr_szsz.o : $(DWALLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DBITREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSZSZ -DFULLDIAGONALIZATION -c -o $@ $<

######

2DWSOURCES= exact.C global.h timer.h samplingcounter.h sparsematrix.h lanczos.h observables.h dwmodel.h inttostring.h statespace.h twodomainstatesmanip.h xyzhamiltonian1d.h sparsematrix.h operator.h spluss.h sx.h sy.h sz.h projectionoperator.h hash.h RunParameter.h observablesspec.h

2dw_corr_sxsx.o : $(2DWSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DDOMAINREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSXSX -c -o $@ $<

2dw_full_corr_sxsx.o : $(2DWSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DDOMAINREPRESENTATION -DDOMAINWALLMODEL -DCORRELATIONFUNCTION -DSXSX -DFULLDIAGONALIZATION -c -o $@ $<



######

OBSERVABLESSOURCES = observables_std.C observables_std.h observables.h lanczos.h sparsematrix.h alltemperatures.h onetemperature.h global.h lowlevelroutines.h observablesspec.h

observables_std.o : $(OBSERVABLESSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

observables_std_corr.o : $(OBSERVABLESSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -c -o $@ $<

OBSERVABLESSOURCES1 = observables_std.C observables_std.h observables.h lanczos.h sparsematrix.h alltemperatures.h onetemperature.h global.h lowlevelroutines.h observablesspec.h

observables_std_oneT.o : $(OBSERVABLESSOURCES1) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DONETEMPERATURE -c -o $@ $<

observables_std_corr_oneT.o : $(OBSERVABLESSOURCES1) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -DONETEMPERATURE -c -o $@ $<


OBSERVABLESFULLSOURCES = observables_std.C observables_std.h observables.h fulldiagonalization.h sparsematrix.h alltemperatures.h global.h lowlevelroutines.h observablesspec.h

observables_std_full.o : $(OBSERVABLESFULLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DFULLDIAGONALIZATION -c -o $@ $<

observables_std_full_corr.o : $(OBSERVABLESFULLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -DFULLDIAGONALIZATION -c -o $@ $<


OBSERVABLESFULLSOURCES1 = observables_std.C observables_std.h observables.h fulldiagonalization.h sparsematrix.h onetemperature.h global.h lowlevelroutines.h observablesspec.h

observables_std_full_oneT.o : $(OBSERVABLESFULLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DFULLDIAGONALIZATION -DONETEMPERATURE -c -o $@ $<

observables_std_full_corr_oneT.o : $(OBSERVABLESFULLSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -DFULLDIAGONALIZATION -DONETEMPERATURE -c -o $@ $<

#####
CUDAOBSERVABLESSOURCES = observables_cuda.cu observables_cuda.h observables.h global.h sparsematrix.h alltemperatures.h lowlevelroutines.h cuda_routines.h cuda_kernels.h cuda_global.h observablesspec.h 

observables_cuda.o:  $(CUDAOBSERVABLESSOURCES) $(SYSTEM)  
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -c -o $@ $<	


observables_cuda_corr.o:  $(CUDAOBSERVABLESSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -c -o $@ $<	


CUDAOBSERVABLESSOURCES1 = observables_cuda.cu observables_cuda.h observables.h global.h sparsematrix.h onetemperature.h lowlevelroutines.h cuda_routines.h cuda_kernels.h cuda_global.h observablesspec.h 

observables_cuda_oneT.o:  $(CUDAOBSERVABLESSOURCES1) $(SYSTEM)  
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -DONETEMPERATURE -c -o $@ $<	


observables_cuda_corr_oneT.o:  $(CUDAOBSERVABLESSOURCES1) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -DCORRELATIONFUNCTION -DONETEMPERATURE -c -o $@ $<	

#####

OBSERVABLESSPECSOURCES = observablesspec.C observablesspec.h

observablesspec.o : $(OBSERVABLESSPECSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DONETEMPERATURE -c -o $@ $<


#####


INTTOSTRINGSOURCES = inttostring.C inttostring.h

inttostring.o : $(INTTOSTRINGSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


#####
SPARSEMATRIXSOURCES = sparsematrix.C sparsematrix.h global.h

sparsematrix.o : $(SPARSEMATRIXSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


#####

# REORTHOGONALIZE uses the Modified Gram-Schmidt process to ensure orthogonality
# MGSREPEATSTEPS=1, Modified Gram-Schmidt only
# MGSREPEATSTEPS=2, Modified Gram-Schmidt with reorthogonalization


LANCZOSSOURCES = lanczos_std.C lanczos_std.h lanczos.h lowlevelroutines.h banddiag.h rnddef.h global.h

lanczos_std.o : $(LANCZOSSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -DREORTHOGONALIZE -DMGSREPEATSTEPS=2 -c -o $@ $<

#####

CUDALANCZOSSOURCES = lanczos_cuda.cu lanczos_cuda.h lanczos.h cuda_global.h cuda_precision.h precision_types.h banddiag.h rnddef.h global.h

lanczos_cuda.o:  $(CUDALANCZOSSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -DREORTHOGONALIZE -DMGSREPEATSTEPS=2 -c -o $@ $<	

#####

OPERATORSOURCES = operator.C operator.h sparsematrix.h inttostring.h global.h

operator.o : $(OPERATORSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

#####

LOWLEVELROUTINESSOURCES= lowlevelroutines.C lowlevelroutines.h

lowlevelroutines.o : $(LOWLEVELROUTINESSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

#####

FULLDIAGONALIZATIONSOURCES = fulldiagonalization.C fulldiagonalization.h

fulldiagonalization.o : $(FULLDIAGONALIZATIONSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

#####

BANDDIAGSOURCES = banddiag.C banddiag.h

banddiag.o : $(BANDDIAGSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

#####

CUDAGLOBALSOURCES = cuda_global.cu cuda_global.h

cuda_global.o : $(CUDAGLOBALSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -c -o $@ $<		

#####

CUDAROUTINESSOURCES = cuda_routines.cu cuda_routines.h cuda_kernels.h 

cuda_routines.o : $(CUDAROUTINESSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -c -o $@ $<		

#####

CUDAKERNELSOURCES = cuda_kernels.cu cuda_kernels.h

cuda_kernels.o : $(CUDAKERNELSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -c -o $@ $<		

#####

CUDAPRECISIONSOURCES = cuda_precision.cu cuda_precision.h cuda_precision_kernels.h precision_types.h

cuda_precision.o : $(CUDAPRECISIONSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) --fmad=false $(GLOBALFLAGS) -c -o $@ $<		

#####

CUDAPRECISIONKERNELSOURCES = cuda_precision_kernels.cu cuda_precision_kernels.h

cuda_precision_kernels.o : $(CUDAPRECISIONKERNELSOURCES) $(SYSTEM)
	$(CUDACC) $(CUDAFLAGS) --fmad=false $(GLOBALFLAGS) -c -o $@ $<		

#####


#CUDAPRECISIONDRIVERSSOURCES = cuda_precision_drivers.cu cuda_precision_drivers.h cuda_precision.h cuda_global.h
#
#cuda_precision_drivers.o : $(CUDAPRECISIONDRIVERSSOURCES) $(SYSTEM)
#	$(CUDACC) $(CUDAFLAGS) $(GLOBALFLAGS) -c -o $@ $<		

#####


ALLTSOURCES  = computediag_allT.C alltemperatures.h 

computediag_allT.o : $(ALLTSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


ONETSOURCES  = computediag_oneT.C onetemperature.h

computediag_oneT.o : $(ONETSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


#####

CORRALLTSOURCES  = computecorr_allT.C alltemperatures.h 

computecorr_allT.o : $(CORRALLTSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

CORRONETSOURCES  = computecorr_oneT.C onetemperature.h

computecorr_oneT.o : $(CORRONETSOURCES) $(SYSTEM)
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


#####

dottest.o : dottest.C
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

testortho.o : testortho.C lowlevelroutines.h
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

gqdtest.o : gqdtest.C cuda_precision.h cuda_global.h
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

doublerandomtest.o : doublerandom.C cuda_precision.h cuda_global.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

doubletest.o : double.C cuda_precision.h cuda_global.h precision_types.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

complexrandomtest.o : complexrandom.C cuda_precision.h cuda_global.h precision_types.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

norm.o : norm.C cuda_precision.h cuda_global.h precision_types.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

complextest.o : complexdouble.C cuda_precision.h cuda_global.h precision_types.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<

Zaxpytest.o : Zaxpy.C cuda_precision.h cuda_global.h precision_types.h makefile
	$(CCC) $(CCFLAGS) $(GLOBALFLAGS) -c -o $@ $<


spotless:	
	make clean
	rm -f *.ps
	rm -rf op_*
	rm -rf sectorstosample
	rm -rf log.txt

clean	:
	rm -f *.dvi
	rm -f *.aux
	rm -f *.log
	rm -f *~
	rm -f core
	rm -f *.o
	rm -f *.exec
	rm -f *.d
	rm -rf *.dat

dataclean:
	rm -rf *.dat








