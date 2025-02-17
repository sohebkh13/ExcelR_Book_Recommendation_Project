import nbformat

# Load the notebook
with open("book_recommendation.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=5)

# Convert and save it as nbformat v4
nbformat.write(nb, "book_recommendation_v4.ipynb", version=4)

print("Notebook successfully converted to nbformat v4!")
