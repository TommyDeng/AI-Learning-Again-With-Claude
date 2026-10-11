from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="models/imdb-distilbert-final",
    tokenizer="models/imdb-distilbert-final",
)

reviews = [
    "A stunning film with brilliant performances. I was moved to tears.",
    "Two hours of my life I will never get back. Dull and predictable.",
    "The first half is slow, but the ending completely redeems it.",
]
for r, out in zip(reviews, clf(reviews)):
    print(f"{out['label']:8} {out['score']:.3f}  {r}")
