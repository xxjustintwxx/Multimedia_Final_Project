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

Run from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error `
  -outdir=build Final_Hand_Gesture_Report_Team14.tex
Copy-Item build\Final_Hand_Gesture_Report_Team14.pdf .
```

## Before Submission

Complete the student IDs and specific member contributions on the final page.

The main text currently ends on page 5. References and the contribution table
are on pages 6 and 7 and are excluded from the 10-page main-text limit.
