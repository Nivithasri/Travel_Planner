# D69: plain vs contextual vs control (dev, nDCG@10)

| family | arm | overall | named | free | model-written only |
|---|---|---|---|---|---|
| dense | plain | 0.701 | 0.771 | 0.657 | 0.800 |
| dense | contextual | 0.829 | 0.867 | 0.805 | 0.822 |
| dense | control | 0.725 | 0.841 | 0.652 | 0.780 |
| bm25 | plain | 0.542 | 0.510 | 0.563 | 0.519 |
| bm25 | contextual | 0.571 | 0.563 | 0.576 | 0.530 |
| bm25 | control | 0.551 | 0.531 | 0.564 | 0.535 |

dense: contextual - plain = +0.128; control - plain = +0.024; contextual - control = +0.104 (the note beyond the place name)

bm25: contextual - plain = +0.028; control - plain = +0.009; contextual - control = +0.019 (the note beyond the place name)

model-written only (c): the retriever keeps provenance=llm rows and is scored against qrels restricted to llm chunks, so it is the same arm on the model-written sub-corpus, not a like-for-like row (D69). 13 dev queries; descriptive (D92).
