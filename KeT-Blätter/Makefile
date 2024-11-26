SUBDIRS := $(dir $(wildcard ./tutorial*/)) $(dir $(wildcard ./exam*/))

all: $(SUBDIRS)

$(SUBDIRS): | build 
	$(MAKE) -C $@

clean:
	rm -rf ./tutorial*/build/
	rm -rf ./exam*/build/
	rm -rf build 

build:
	mkdir -p build 

.PHONY: all $(SUBDIRS)
