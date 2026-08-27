# QAT-VQ Paper

`qatvq_paper.tex` + `references.bib` — draft paper synthesizing all three
experiment branches (DistilBERT/SST-2, GPT-2/WikiText-2, GPT-2 scaling
study) into one submission-ready document.

Checked for structural correctness (balanced braces, matched
`\begin`/`\end`, consistent table column counts, all citations resolve to
`references.bib` entries) but **not yet compiled** — no LaTeX toolchain was
available in the environment it was written in. Compile before submitting.

## Compile

Easiest: paste both files into a new project on [Overleaf](https://overleaf.com)
(free) and hit Recompile.

Locally, with a LaTeX distribution installed (e.g. `sudo apt install
texlive-latex-base texlive-latex-extra texlive-bibtex-extra`):

```bash
pdflatex qatvq_paper.tex
bibtex qatvq_paper
pdflatex qatvq_paper.tex
pdflatex qatvq_paper.tex   # twice more to resolve references/citations
```

## Before submitting to arXiv

1. Compile and proofread the PDF.
2. Fill in any co-authors (e.g. your thesis supervisor, if appropriate).
3. Get arXiv endorsement in `cs.LG` (primary) — ask your supervisor if
   they've published there before.
4. Submit source (`.tex` + `.bib`, not just the PDF) so arXiv can typeset
   it natively.

Full result tables, figures, and reproduction commands for every number in
this paper are in the three experiment branches:
[`distilbert-sst2-qatvq`](../../tree/distilbert-sst2-qatvq),
[`gpt2-wikitext2-qatvq`](../../tree/gpt2-wikitext2-qatvq),
[`gpt2medium-wikitext103-qatvq`](../../tree/gpt2medium-wikitext103-qatvq).
