all: build/thesis.pdf

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
#build/thesis.pdf: gmlimit.py | build
#	python gmlimit.py

build/thesis.pdf: build/g_1m_1test.pdf

build/thesis.pdf: FORCE | build
	python code/gmlimit.py
	latexmk $(TeXOptions) thesis.tex 
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
