all: build/V206.pdf

# hier Python-Skripte:
build/plot.pdf: plot.py ../matplotlibrc ../header-matplotlib.tex | build
	# so that matplotlib can find the tex header when running
	# LaTeX in the tmp directory
	# and set the matplotlibrc
	TEXINPUTS=$$(pwd)/..: MATPLOTLIBRC=../matplotlibrc python plot.py

# hier weitere Abhängigkeiten für build/vXXX.pdf deklarieren:
build/V206.pdf: build/plot.pdf build/plot_1.pdf build/plot_2.pdf build/plot_3.pdf build/plot_4.pdf build/plot_5.pdf 

build/V206.pdf: FORCE | build
	# to find header and bib files in the main directory
	TEXINPUTS=..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V206.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
