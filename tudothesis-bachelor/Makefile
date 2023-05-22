all: build/thesis.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
#build/thesis.pdf: gmlimit.py | build
#	python gmlimit.py

build/thesis.pdf: FORCE | build
	python code/g_eegmlimit.py
	python code/g_eevis.py
	python code/gmlimitLMA.py
	python code/gmlimitSMA.py
	python code/gmlimitLOW.py
	python code/gmlimitVAC.py
	latexmk $(TeXOptions) thesis.tex 

	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
