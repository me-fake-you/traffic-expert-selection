# Historical source archive

Files under `original/` are byte-identical copies of the experiment scripts listed in `SOURCE_MANIFEST.csv`. They preserve the original settings and execution order. Local directory conventions inside them are intentionally not rewritten and are not new user-facing commands.

This is a selective source archive, not the whole thesis project. It does not contain credential-handling modules, downloaded third-party projects, raw traffic, checkpoints, or the private run history. Some imports and paths refer to the original workspace, including upstream preparation, OOF/reliability records and traffic adapters. Running an archived script directly in this new repository is not a supported reproduction path and may fail for missing inputs. No complete-from-PCAP reproduction claim is made for this release candidate.

Use `traffic_selection` for the portable frozen-prediction analysis. Use these files to inspect how the original training, controls, perturbations and execution were implemented. Reconstructing the full original environment is a separate, data-rights-dependent packaging task, not an additional experiment performed for this release.

The source manifest's relative historical paths are provenance labels. They are not links to public datasets or a claim that those result directories have been published.
