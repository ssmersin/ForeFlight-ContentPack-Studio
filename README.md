# ForeFlight Content Pack Studio

Windows-first Python project for analyzing aviation PDF publications and preparing ForeFlight content-pack data.

## Open the project

Open \`ForeFlight-ContentPack-Studio.sln\` in Visual Studio 2026. The solution contains the Python project in \`App/\`; install \`App/requirements.txt\` into the selected Python environment.

## Current milestone

The application reads the document type and effective date range from the PDF cover, classifies chart pages, writes a CSV manifest, and exports each detected chart page as an individual one-page PDF under \`HEDEF/Initial/\`.

Ordinary PDFs are exported as optimized vector PDFs. GeoPDFs are detected and copied through a separate object-preserving path that retains page viewport/measure dictionaries and catalog-level \`/LGIDict\` data; real-world GeoPDF samples are still required for broader validation. No user interface is being added at this stage.

No branches are created without the repository owner's approval; development proceeds directly on \`main\`.
