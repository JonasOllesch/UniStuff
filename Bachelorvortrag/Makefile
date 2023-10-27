all: build/bavortrag.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/bavortrag.pdf: FORCE | build
	latexmk $(texoptions) bavortrag.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc bavortrag.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
