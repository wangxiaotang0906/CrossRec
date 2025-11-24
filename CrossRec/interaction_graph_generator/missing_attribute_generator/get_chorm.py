import scanpy as sc
import pandas as pd
import requests
from tqdm import tqdm

rna_anndata = sc.read_h5ad("../../../bio_datasets/NeurIPS2021_Multiome/neurips2021_multiome_scRNA.h5ad")
print("RNA anndata:", rna_anndata)  # 69249 cells, 13431 genes
print(rna_anndata.var)
atac_anndata = sc.read_h5ad("../../../bio_datasets/NeurIPS2021_Multiome/neurips2021_multiome_scATAC.h5ad")
print("ATAC anndata:", atac_anndata)  # 69249 cells, 116490 peaks
print(atac_anndata.var)

def parse_peak(peak):
    if not isinstance(peak, str):
        return None, None, None
    # typical format: "chr1-12345-12456"
    parts = peak.split("-")
    if len(parts) != 3:
        return None, None, None
    chrom = parts[0]
    try:
        start = int(parts[1])
        end = int(parts[2])
    except ValueError:
        return None, None, None
    return chrom, start, end

atac_var = atac_anndata.var.copy()
chrom_list, start_list, end_list = [], [], []

for peak in atac_var.index:
    chrom, st, ed = parse_peak(peak)
    chrom_list.append(chrom)
    start_list.append(st)
    end_list.append(ed)

atac_var["chrom"] = chrom_list
atac_var["start"] = start_list
atac_var["end"] = end_list
atac_anndata.var = atac_var
print(atac_anndata.var)

# GTF gene_name → chrom
def load_gtf_gene_id_to_chrom(gtf_path):
    gene_chrom = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.split("\t")
            if fields[2] != "gene": continue
            chrom = fields[0]
            attr = fields[8]
            info = {}
            for kv in attr.split(";"):
                kv = kv.strip()
                if " " in kv:
                    k, v = kv.split(" ", 1)
                    info[k] = v.strip('"')
            gid = info.get("gene_id")
            if gid:
                # Delete version: ENSG00000290825.1 → ENSG00000290825
                gid = gid.split(".")[0]
                gene_chrom[gid] = chrom

    return gene_chrom

gtf_path = "gencode.v43.annotation.gtf"
rna_gene_ids = rna_anndata.var["gene_id"].str.split(".").str[0]
chrom_map = load_gtf_gene_id_to_chrom(gtf_path)
rna_anndata.var["chrom"] = rna_gene_ids.map(chrom_map).fillna("chrUnknown")

print(rna_anndata.var)

# 保存
rna_anndata.write_h5ad("rna_with_chr.h5ad")
atac_anndata.write_h5ad("atac_with_chr.h5ad")

print("Done.")
