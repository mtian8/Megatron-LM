from megatron.core.datasets import indexed_dataset
import numpy as np

# idx_file = '/u/mtian8/LLM/data/software/software_text_document.idx'
idx_file = '/u/mtian8/LLM/data/pub_ocr/pub_ocr_text_document.idx'
IR = indexed_dataset._IndexReader(idx_file,False)
sum = 0
for i in range(IR.__len__()):
    sum +=IR.__getitem__(i)[1]
print(sum)
    