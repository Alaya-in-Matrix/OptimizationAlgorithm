#!/bin/bash
# pandoc report.md \
#        -f markdown_github \
#        -t latex \
#        --latex-engine=xelatex \
#        --template=template.tex \
#        -S \
#        -V mainfont=Hei \
#        -o test.latex

pandoc report.md -f markdown_github -o test.docx

pandoc -D html > tmp.html
pandoc report.md -f markdown_github -o test.html --template=tmp.html
rm tmp.html
