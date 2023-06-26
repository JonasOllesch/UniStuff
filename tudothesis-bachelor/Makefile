all: build/thesis.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
#build/thesis.pdf: gmlimit.py | build
#	python gmlimit.py


build/thesis.pdf: FORCE | build
	python code/finalplot.py
	python code/g_eegmlimit.py
	python code/gmlimitnew.py
#	python unnötiger_code/gmlimitlowernew.py
#	python unnötiger_code/g_eevis.py
#	python unnötiger_code/gmlimitLMA.py
#	python unnötiger_code/justgmlimits.py
#	python unnötiger_code/gmlimitSMA.py
#	python unnötiger_code/gmlimitLOW.py
#	python unnötiger_code/gmlimitVAC.py
	latexmk $(TeXOptions) thesis.tex 

	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
