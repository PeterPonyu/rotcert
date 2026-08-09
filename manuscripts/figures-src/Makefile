# Build the rotcert manuscript figures: frozen result JSON -> CSV -> standalone
# pgfplots PDF. The submission .tex files include ONLY the compiled PDFs via
# \includegraphics (no inline tikzpicture in the paper).
#
# Usage:
#   make            # regenerate CSVs from frozen JSONs, then compile both PDFs
#   make data       # regenerate CSVs only
#   make clean      # remove LaTeX build byproducts (keeps PDFs and CSVs)
#   make distclean  # also remove the compiled PDFs and generated CSVs

LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

FIGS = fig0_overview fig1_diorperclass fig2_covmatched
CSVS = fig1_perclass.csv fig1_series_ORCNN.csv fig1_series_RTMDet.csv \
       fig1_series_RoI-Trans.csv fig1_series_S2A-Net.csv fig1_band.csv \
       fig2_covmatched.csv fig2_series_naivecoord.csv fig2_series_hull.csv

.PHONY: all data clean distclean

all: data $(addsuffix .pdf,$(FIGS))

data: $(CSVS)

$(CSVS): extract_data.py
	python3 extract_data.py

# fig0_overview is a pure symbolic pipeline schematic (no result data; no
# CSV/extract_data.py dependency), unlike fig1/fig2 below.
fig0_overview.pdf: fig0_overview.tex
	$(LATEXMK) fig0_overview.tex

fig1_diorperclass.pdf: fig1_diorperclass.tex fig1_series_ORCNN.csv fig1_series_RTMDet.csv fig1_series_RoI-Trans.csv fig1_series_S2A-Net.csv fig1_band.csv
	$(LATEXMK) fig1_diorperclass.tex

fig2_covmatched.pdf: fig2_covmatched.tex fig2_series_naivecoord.csv fig2_series_hull.csv
	$(LATEXMK) fig2_covmatched.tex

clean:
	latexmk -c fig0_overview.tex
	latexmk -c fig1_diorperclass.tex
	latexmk -c fig2_covmatched.tex

distclean: clean
	rm -f $(addsuffix .pdf,$(FIGS)) $(CSVS)
