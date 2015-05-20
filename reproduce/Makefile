all:
	./ipnbdoctest.py results.ipynb temp.ipynb
	ipython nbconvert --to html --template basic temp.ipynb
	mv temp.html ../_includes/notebook.html
	rm temp.ipynb
