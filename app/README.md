# Desktop application

The graphical application described in Appendix B of the first version of the paper
(Section S10 of the supplementary material of the revised version) is not part of this
first release of the repository, and the reason is worth stating plainly rather than
leaving it implicit.

It is a convenience wrapper. Everything the paper reports is produced by the package
and the scripts in this repository, and none of it goes through the application. What
the application adds is a way for an experimentalist to run the workflow without
touching a command line: load a micrograph, pick the analysis mode, see the overlay,
export the JSON report, or run a whole campaign in batch.

Two things keep it out of this release. Its interface is written in Spanish, having
been built for use in the laboratory at CIEMAT, and it is coupled to the flat file
layout that preceded this package, invoking the training and inference scripts as
subprocesses by their old file names. Translating the interface and rewiring it to the
package are both straightforward, but neither can be verified without running the
application against a real display, so shipping it untested here would be worse than
not shipping it.

It will be added once that work is done and tested. In the meantime the compiled
Windows executable is available from the corresponding author, and the same
functionality is reachable from the command line:

```bash
# single image or whole directory, with the JSON report and the annotated figure
python -m tungsten_analyzer.inference --input ./data/images --output ./inference_results
```
