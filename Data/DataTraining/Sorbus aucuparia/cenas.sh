for f in *.png; do 
mv -- "$f" "${f%.png}.jpg"
done