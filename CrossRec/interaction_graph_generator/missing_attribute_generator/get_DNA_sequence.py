from pyfaidx import Fasta
import scanpy as sc

atac_anndata = sc.read_h5ad("../../../bio_datasets/ISSAAC-seq/ISSAAC_scATAC_format.h5ad")
print('atac_anndata:',atac_anndata)
print('atac_anndata.var:',atac_anndata.var)
peak_df = atac_anndata.var.copy()  # chrom, chromStart, chromEnd）

fasta = Fasta("hg38.fa")


out_fa_path = "bio_datasets/ISSAAC-seq/ISSAAC_peak_output"

def fetch_dna(chrom, start, end):
    if chrom not in fasta:
        return "N" * (end - start)
    try:
        seq = fasta[chrom][start:end].seq.upper()
    except Exception:
        seq = "N" * (end - start)
    return seq

sequences = []
peak_names = []
for chrom, s, e in zip(peak_df["chrom"], peak_df["chromStart"], peak_df["chromEnd"]):
    peak_id = f"{chrom}:{s}-{e}"
    peak_names.append(peak_id)
    seq = fetch_dna(chrom, int(s), int(e))
    target_len = 256
    if len(seq) >= target_len:
        seq = seq[:target_len]
    else:
        seq = seq + "N" * (target_len - len(seq))
    sequences.append(seq)
print("Total peaks:", len(sequences))

with open(out_fa_path, "w") as f:
    for name, seq in zip(peak_names, sequences):
        f.write(f">{name}\n")
        f.write(seq + "\n")

print("Saved:", out_fa_path)
