all: build/report.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/report.pdf: FORCE | build
	latexmk $(TeXOptions) report.tex
	
FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
