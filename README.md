# CON2PHYS

CON2PHYS is a Python project for loading and analysing consortium neuroscience data, including spike trains, local field potentials (LFPs), trial events, and recording metadata. It contains a reusable Pynapple-based data loader and a collection of analysis scripts covering firing rates, statistical comparisons, visualisations, and dimensionality-reduction methods such as PCA and tensor component analysis.

Install the dependencies with `pip install -r requirements.txt`, place the dataset (for example, `py-files.zip`) in the `data/` directory, and run an analysis from the repository root, such as `python src/Q1.py`. Tests can be run with `pytest`.
