all: build/V308.pdf

# hier Python-Skripte:
build/plot.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/V308.pdf deklarieren:
build/V308.pdf: build/plot.pdf build/Hystereseplot.pdf build/langeSpule.pdf build/SpulenPaar10.pdf build/SpulenPaar15.pdf build/SpulenPaar20.pdf

build/V308.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V308.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
