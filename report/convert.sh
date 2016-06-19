#!/bin/bash
# pandoc report.md \
#        -f markdown \
#        -t latex \
#        --latex-engine=xelatex \
#        --template=template.tex \
#        -S \
#        -V mainfont="Noto Sans CJK SC" \
#        -V geometry:margin=1in \
#        -o test.latex
echo pdf
pandoc report.md \
       -f markdown \
       -t latex \
       --latex-engine=xelatex \
       --template=template.tex \
       -S \
       -V mainfont="AR PL UKai CN" \
       -V geometry:margin=1in \
       --toc \
       -o test.pdf
echo docx
pandoc report.md \
       -f markdown -t docx \
       -o test.docx
echo html
pandoc report.md \
       -f markdown -t html \
       --template=template.html \
       -V mainfont="Noto Sans CJK SC" \
       --mathjax=./MathJax-master/MathJax.js?config=TeX-AMS-MML_HTMLorMML \
       -o test.html
