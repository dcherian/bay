fname=paper1

# org2latex $fname.org

hash=$(git log -n 1 --pretty=%H)

pdflatexstr='pdflatex --shell-escape %O %S && exiftool -overwrite_original -Producer='$hash' %B.pdf'
echo $pdflatexstr

if [ $? == 0 ]; then
    latexmk -gg -c supplementary.tex
    latexmk -gg -c -pvc -latex="$pdflatexstr"  -pdf $fname.tex;
fi
exiftool -overwrite_original -Producer='$hash' $fname.pdf

if xdotool search --name "$fname.pdf -" key r; then
    echo "PDF updated."
else
   exo-open $fname.pdf
   echo "PDF opened."
fi
