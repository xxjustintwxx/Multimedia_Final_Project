# Team 14 Final Report

## Files

- `Final_Hand_Gesture_Report_Team14.tex`: report source
- `Final_Hand_Gesture_Report_Team14.pdf`: compiled report
- `references.bib`: bibliography
- `figures/`: figures used by the report
- `source/`: final presentation and official challenge specification
- `evidence/`: helper code for verifying quantitative claims
- `build/`: generated LaTeX files and visual-QA previews (ignored by Git)

## Compile

### Windows (PowerShell)

Run from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error `
  -outdir=build Final_Hand_Gesture_Report_Team14.tex
Copy-Item build\Final_Hand_Gesture_Report_Team14.pdf .
```

### Linux (bash)

Install a TeX Live with XeLaTeX once (Fedora):

```bash
sudo dnf install -y texlive-scheme-medium texlive-xetex latexmk \
  texlive-pgf texlive-ieeetran texlive-collection-fontsrecommended texlive-lm
```

Then compile from this directory:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=build Final_Hand_Gesture_Report_Team14.tex
cp build/Final_Hand_Gesture_Report_Team14.pdf .
```

### Chinese font note

The report uses `Microsoft JhengHei` when present (Windows). On systems
without it (e.g. Linux), it falls back automatically to `Noto Sans CJK TC`
(`google-noto-sans-cjk-vf-fonts` on Fedora), so the member names still render.

## Before Submission

Complete the student IDs and specific member contributions on the final page.

The main text currently ends on page 5. References and the contribution table
are on pages 6 and 7 and are excluded from the 10-page main-text limit.
