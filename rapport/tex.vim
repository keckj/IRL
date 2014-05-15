"
"Bind des touches pour les fichiers *.tex
"
map <F5> :!clear && pdflatex %<cr>
map <F6> :!clear && pdflatex % && bibtex %:r.aux && pdflatex % && pdflatex %<cr>
map <F7> :!evince %:p:r.pdf &<cr>

