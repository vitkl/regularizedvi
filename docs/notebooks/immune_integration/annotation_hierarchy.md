# Annotation Hierarchy

Unified hierarchy for all harmonized cell type names across all datasets.
`level_1` is the finest resolution (= harmonized_name for unique types), `level_4` is the coarsest.

| harmonized_name           | level_1             | level_2           | level_3                | level_4               |
| ------------------------- | ------------------- | ----------------- | ---------------------- | --------------------- |
| **HSCs & Progenitors**    |                     |                   |                        |                       |
| HSC                       | HSCs                | HSCs              | HSCs                   | Hematopoietic lineage |
| Lymph prog                | Lymph prog          | Lymph prog        | Lymphoid lineage       | Hematopoietic lineage |
| G/M prog                  | G/M prog            | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| MK/E prog                 | MK/E prog           | MK/E prog         | Erythroid / MK lineage | Hematopoietic lineage |
| ID2-hi myeloid prog       | ID2-hi myeloid prog | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| **Erythroid**             |                     |                   |                        |                       |
| Proerythroblast           | Erythroid lineage   | Erythroid lineage | Erythroid / MK lineage | Hematopoietic lineage |
| Erythroblast              | Erythroid lineage   | Erythroid lineage | Erythroid / MK lineage | Hematopoietic lineage |
| Normoblast                | Erythroid lineage   | Erythroid lineage | Erythroid / MK lineage | Hematopoietic lineage |
| **CD4+ T cells**          |                     |                   |                        |                       |
| CD4+ T naive              | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T activated          | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T                    | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T central memory     | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T effector memory    | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T CTL                | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T uncommitted memory | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T Th1 memory         | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T Th2 memory         | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T Th17 memory        | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD4+ T Th1/Th17 memory    | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Treg                      | CD4+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| **CD8+ T cells**          |                     |                   |                        |                       |
| CD8+ T                    | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD8+ T activated          | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD8+ T naive              | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD8+ T effector memory    | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD8+ T central memory     | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| CD8+ T proliferating      | CD8+ T-cell lineage | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| **Other T cells**         |                     |                   |                        |                       |
| T central memory          | T central memory    | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| MAIT                      | MAIT                | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| gamma-delta T             | gamma-delta T       | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| double-negative T         | double-negative T   | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| IFN-responding T          | IFN-responding T    | T-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| **NK / ILC**              |                     |                   |                        |                       |
| NK                        | NK-cell lineage     | NK-cell lineage   | Lymphoid lineage       | Hematopoietic lineage |
| NK CD56bright             | NK-cell lineage     | NK-cell lineage   | Lymphoid lineage       | Hematopoietic lineage |
| NK proliferating          | NK-cell lineage     | NK-cell lineage   | Lymphoid lineage       | Hematopoietic lineage |
| NK TGFB1+                 | NK-cell lineage     | NK-cell lineage   | Lymphoid lineage       | Hematopoietic lineage |
| ILC                       | NK-cell lineage     | NK-cell lineage   | Lymphoid lineage       | Hematopoietic lineage |
| **B cells**               |                     |                   |                        |                       |
| Naive B                   | Naive B             | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Transitional B            | Developing B        | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Intermediate B            | Developing B        | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| B1 B                      | B1 B                | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Memory B                  | Memory B            | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Activated B               | Activated B         | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| Plasma B cell             | Plasma B cell       | B-cell lineage    | Lymphoid lineage       | Hematopoietic lineage |
| **Myeloid / DC**          |                     |                   |                        |                       |
| CD14+ Mono                | Monocyte lineage    | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| CD16+ Mono                | Monocyte lineage    | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| Proinflammatory Mono      | Monocyte lineage    | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| cDC                       | DC lineage          | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| cDC2                      | DC lineage          | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| pDC                       | DC lineage          | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| ASDC                      | DC lineage          | Myeloid lineage   | Myeloid lineage        | Hematopoietic lineage |
| **Other**                 |                     |                   |                        |                       |
| Platelet                  | Platelet            | Platelet          | Erythroid / MK lineage | Hematopoietic lineage |
