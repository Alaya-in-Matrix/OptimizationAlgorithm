#!/bin/bash
pandoc report.md \
       --latex-engine=xelatex \
       --template=template.tex \
       -S \
       -V mainfont=Hei \
       -o test.pdf
pandoc report.md -o test.docx
pandoc -D html > template.html
# pandoc report.md --template=template.html --mathjax=https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML -o test.html
pandoc report.md --template=template.html --mathjax=./MathJax-master/MathJax.js?config=TeX-AMS-MML_HTMLorMML -o test.html
