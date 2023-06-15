all: build/thesis.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
#build/thesis.pdf: gmlimit.py | build
#	python gmlimit.py


build/thesis.pdf: FORCE | build
	python code/finalplot.py
	python code/gmlimitnew.py
#	python code/gmlimitlowernew.py
#	python code/g_eegmlimit.py
#	python code/g_eevis.py
#	python code/gmlimitLMA.py
#	python code/justgmlimits.py
#	python unnötiger_code/gmlimitSMA.py
#	python unnötiger_code/gmlimitLOW.py
#	python unnötiger_code/gmlimitVAC.py
	latexmk $(TeXOptions) thesis.tex 

	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
