LATEX ?= lualatex
TARGET ?= presentation
FIGURES := $(wildcard Figures/*.png)
TEXCACHE ?= $(CURDIR)/.texcache

export TEXMFVAR := $(TEXCACHE)/texmf-var
export TEXMFCONFIG := $(TEXCACHE)/texmf-config
export TEXMFHOME := $(TEXCACHE)/texmf-home
export LUAOTFLOAD_CACHE := $(TEXCACHE)/luaotfload

.PHONY: all clean distclean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex beamerthemeSimple.sty $(FIGURES)
	mkdir -p $(TEXMFVAR) $(TEXMFCONFIG) $(TEXMFHOME) $(LUAOTFLOAD_CACHE)
	$(LATEX) -interaction=nonstopmode $(TARGET).tex
	$(LATEX) -interaction=nonstopmode $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).nav $(TARGET).out
	rm -f $(TARGET).snm $(TARGET).toc $(TARGET).vrb
	rm -f $(TARGET).synctex.gz

distclean: clean
	rm -f $(TARGET).pdf
	rm -rf $(TEXCACHE)
