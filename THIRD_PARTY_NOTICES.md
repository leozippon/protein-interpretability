# Licensing scope and third-party notices

The [MIT license](LICENSE.md) covers this project's original software, including its measurement library, scripts, tests and accompanying software documentation. It does not replace the licenses of third-party components, confer rights in downloaded resources, or apply automatically to scientific manuscripts, their prose, figures or data. A separate license stated for a file or release takes precedence for that material.

## Springer Nature manuscript support files

The local, unpublished manuscript uses unchanged `sn-jnl.cls` and `sn-nature.bst` files from the [official Springer Nature journal template](https://www.springernature.com/gp/authors/campaigns/latex-author-support). Their original notices remain intact: the class specifies the LaTeX Project Public License (LPPL), version 1.3c or later, and the bibliography style specifies LPPL version 1 or later. The class also carries the LaTeX base-distribution notice. These files are not relicensed under MIT. This code release does not distribute the manuscript or its support files.

The [LaTeX Project](https://www.latex-project.org/get/) supplies the complete LaTeX system and its sources; the [LPPL text](https://www.latex-project.org/lppl/) provides the governing distribution and modification terms. This project does not claim to maintain or provide upstream support for the template files.

## External software and scientific materials

Imported packages, external source checkouts, pretrained models, model weights and biological databases retain their respective upstream licenses and access conditions. The [resource manifest](external_resources/manifests/interpretability_transfer_resources.json) identifies the project's external inputs; it is not a grant to redistribute them. Downloaded payloads remain outside Git. In particular, the project MIT license does not cover ProGen3 or ESMFold weights, the Pfam database, UniRef sequences, or separately staged HMMER and DIAMOND binaries.

## Staged measurement data

Every dataset directory under the ignored `data/` tree carries its source, release, retrieval route, licence and redistribution constraint in `data/dataset_registry.json`, beside a SHA-256 for each staged file. That registry is the per-dataset licence record and this section does not restate it; [the provenance document](docs/DATASET_PROVENANCE.md) explains how it is built and verified.

Attribution under CC-BY-4.0 is required for the AlphaFold Protein Structure Database, Gene Ontology and UniProt-GOA, UniProt Swiss-Prot and UniRef50, ExPASy ENZYME, Pika-DS, the Tsuboyama 2023 MegaScale stability measurements, the Cho MGnify domain stabilities, Domainome 1.0 and SKEMPI 2.0, together with anything derived from them — which includes the k-mer background counts, the EC-labelled FASTA and the Swiss-Prot pickles. InterPro and Pfam are CC0 and carry no attribution condition. Two staged directories have no licence this project can establish: the three curated variant-mechanism tables under `data/mechanism`, whose source articles were not recorded, and the BioLiP ligand-binding table under `data/pdb`, which was staged without its terms. Both are recorded in the registry as unrecoverable and neither may be redistributed until identified.

The Cho MGnify deposit additionally asks users to register their use with the depositors. That is a request rather than a licence condition, and it is recorded so that it is not lost.

The original plotting scripts are MIT-licensed software. Their generated figures, manuscript text and released scientific data require the applicable article or data-release terms; including an artifact in this repository does not by itself assign it the software license. Public access to a cited source or dataset does not replace that source's own attribution and reuse requirements.
