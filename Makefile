.PHONY: all aps mdpi submission clean

TEX_DIR := tex
BUILD_DIR := tex/build

all: aps mdpi

aps:
	cd $(TEX_DIR) && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex

mdpi:
	cd $(TEX_DIR) && latexmk -pdf -interaction=nonstopmode -outdir=build paper_in_MDPI_template.tex

submission: mdpi
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts-and-data/make_submission.ps1

clean:
	cd $(TEX_DIR) && latexmk -C -outdir=build main.tex
	cd $(TEX_DIR) && latexmk -C -outdir=build paper_in_MDPI_template.tex
