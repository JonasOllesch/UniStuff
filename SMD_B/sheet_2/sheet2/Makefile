all: build/sheet2.pdf

# hier Python-Skripte:
build/plot.pdf: code-ex12.py  ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc 
		python code-ex12.py 

# hier weitere Abhängigkeiten für build/sheet2.pdf deklarieren:
build/sheet2.pdf: build/plot.pdf

build/sheet2.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	sheet2.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
