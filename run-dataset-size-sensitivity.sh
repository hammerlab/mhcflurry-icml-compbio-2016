mhcflurry-dataset-size-sensitivity.py \
	--allele HLA-A0201  \
	--training-csv data/bdata.2009.mhci.public.1.txt \
	--imputation-method mice \
	--number-dataset-sizes 15 \
	--random-negative-samples 0 \
	--min-observations-per-peptide 3 \
	--training-epochs 250 \
	--repeat 3 \
	--max-training-samples 500 \
	--min-training-samples 10 \
	--dropout 0.5 \
	--hidden-layer-size 64 \
	--embedding-size 32
#	--load-existing-data

