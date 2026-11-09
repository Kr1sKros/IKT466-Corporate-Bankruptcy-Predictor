# Report section sources

These English Markdown sections preserve the supplied subsection titles. They
are ready for teammate review and transfer into the group's MSc thesis template;
this repository currently contains no thesis template or report build system.
Equations in the GRU section use the PyTorch convention explicitly.

| Issue | Section | Acceptance coverage |
|---|---|---|
| [#19](https://github.com/Kr1sKros/IKT466-Corporate-Bankruptcy-Predictor/issues/19) | [2.0.1 Problem Formulation](sections/02_01_problem_formulation.md) | Fixed company-specific anchor; 18 inputs; failed=1; intended t+1 versus observed status; retrospective and follow-up limitations |
| [#20](https://github.com/Kr1sKros/IKT466-Corporate-Bankruptcy-Predictor/issues/20) | [2.0.2 Classification Models](sections/02_02_classification_models.md) | LR, XGBoost and MLP assumptions/scores; GRU positioning; no expansion of experiment scope |
| [#21](https://github.com/Kr1sKros/IKT466-Corporate-Bankruptcy-Predictor/issues/21) | [2.0.3 GRU and Financial Histories](sections/02_03_gru_financial_histories.md) | Gates/dimensions; k=1-5 at the same anchor; oldest-to-newest recurrence; short histories and gaps without inventing policy |

## Integration and review

- References are attached to each section. Transfer them into the final shared
  bibliography when integrating; section-local numbers are not global BibTeX keys.
- No model scores, trained configurations or completed experiments are claimed.
- #45 must resolve feature meanings and target timing. Keep target_verified false
  unless evidence supports changing it; these sections do not resolve that issue.
- #49 must resolve short-history/missing-year policy and reconcile the current
  newest_to_oldest configuration with the agreed oldest-to-newest input order.
- Human review is still required before merging/closing the three issues.
- AI assistance: Codex drafted these sections and checked them against issue
  acceptance criteria, the committed audit/configuration, and the linked primary
  references. The group must review mathematical notation and substantive claims
  and include the assistance in its final AI-use statement.

Review performed on 9 November 2026: subsection titles, notation across sections,
target caveats, reference destinations, local links and whitespace were checked.
No software behavior was changed; no model-training results were generated.
