export TEXINPUTS := ../:${TEXINPUTS}
CONTENTFILE := $(shell find . -maxdepth 1 -name "exam*.tex")
OUTPUTFILE := $(foreach FILE, $(CONTENTFILE), $(shell basename ${FILE} .tex))
OUTPUTFILE += $(foreach FILE, $(OUTPUTFILE), solution-$(FILE))
OUTPUTFILE := $(foreach FILE, $(OUTPUTFILE), build/$(FILE).pdf)

all: .copyfiles

build/solution-%.tex: %.tex | build
	sed 's/\\def\\solution{1}/\\def\\solution{1}/' $< > $@

build/solution-%.pdf: build/solution-%.tex
	latexmk -lualatex -jobname=$(basename $@) $<

build/%.tex: %.tex | build
	sed 's/\\def\\modelsolution{1}/% \\def\\modelsolution{1}/' $< > $@

build/%.pdf: build/%.tex
	latexmk -lualatex -jobname=$(basename $@) $<

.copyfiles: $(OUTPUTFILE)
	cp $^ build
	touch $^

build:
	mkdir -p build

clean:
	rm -rf build
