=== Description ===

This dataset is MINC-2500

This directory contains a subset of MINC for material classification on single patches, as described in Section 5.4 of the paper. Unlike MINC, this dataset is balanced with 2500 samples per category, each sample sized 362 x 362. Five random splits of 2250 training and 250 testing samples per category are provided.  Each sample comes from a distinct photo (no photo is sampled twice).

The size of each patch is 256*sqrt(2)/1100 times the smaller image dimension. In many cases, patches extend beyond the limits of the image border. Out-of-image pixels were filled with rgb(124,117,104).

MINC is described in

@article{bell15minc,
	author = "Sean Bell and Paul Upchurch and Kavita Bala and Noah Snavely",
	title = "Material Recognition in the Wild with the Materials in Context Database",
	journal = "Computer Vision and Pattern Recognition (CVPR)",
	year = "2015",
}

Please cite this paper if you use this data in a publication.

Contact: sbell@cs.cornell.edu, paulu@cs.cornell.edu

=== Contents ===

categories.txt - An ordered list of categories (zero-indexed labels).

images/CATEGORY/*.jpg - patches for the corresponding category

labels/trainN.txt - List of filenames, N = 1, ..., 5 (different splits).
labels/validateN.txt - Same
labels/testN.txt - Same

Five random splits are provided, each with a 85:5:10 train:validate:test ratio.
