# HotNet_PPR_maker_GPU

A simple Python script to generate PPR matrices to be used with the HotNet2 network propagation pipeline. The most computationallky expensive part of the process is done on a GPU leading to a massive speadup.
Depends on NetworkX, numpy, scipy and cupy (a GPU port of numpy)
