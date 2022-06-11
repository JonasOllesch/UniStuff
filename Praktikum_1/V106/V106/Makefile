all: build/V106.pdf

# hier Python-Skripte:
build/plot.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/V106.pdf deklarieren:
build/V106.pdf: #build/plot.pdf #../v106_abb1.pdf ../v106_abb2.pdf ../v106_abb3.pdf

build/V106.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V106.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
