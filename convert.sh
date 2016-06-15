#!/bin/bash
pandoc README.md \
       -f markdown_github \
       -t latex \
       --latex-engine=xelatex \
       --template=template.tex \
       -V mainfont=Hei \
       -o test.pdf

pandoc README.md -f markdown_github -o test.docx

pandoc -D html > tmp.html
pandoc README.md -f markdown_github -o test.html --template=tmp.html
rm tmp.html
