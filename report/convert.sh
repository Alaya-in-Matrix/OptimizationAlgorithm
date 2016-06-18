#!/bin/bash
pandoc report.md \
       -f markdown \
       -t latex \
       --latex-engine=xelatex \
       --template=template.tex \
       -S \
       -V mainfont="Noto Sans CJK SC" \
       -V geometry:margin=1in \
       -o test.latex
pandoc report.md \
       -f markdown \
       -t latex \
       --latex-engine=xelatex \
       --template=template.tex \
       -S \
       -V mainfont="Noto Sans CJK SC" \
       -V geometry:margin=1in \
       --toc \
       -o test.pdf
pandoc report.md \
       -f markdown -t docx \
       -o test.docx
pandoc report.md \
       -f markdown -t html \
       --template=template.html \
       -V mainfont="Noto Sans CJK SC" \
       --mathjax=./MathJax-master/MathJax.js?config=TeX-AMS-MML_HTMLorMML \
       -o test.html
