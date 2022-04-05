import bbi
import cooler
import pandas as pd
import numpy as np

def chip_loader(regions_df: pd.DataFrame, path: str, resolution: int):
    with bbi.open(path) as f:
        for i, row in regions_df.iterrows():
            signal = f.fetch(row.chrom, row.start, row.end, bins=(row.end-row.start)//resolution)
            yield signal


def hic_loader(regions_df: pd.DataFrame, path: str, resolution: int):
    c = cooler.Cooler(path+'::resolutions/'+str(resolution))
    region_size = regions_df.iloc[0].end-regions_df.iloc[0].start
    dimension = region_size//resolution
    for i, row in regions_df.iterrows():
        
        mat = c.matrix(balance=True).fetch((row.chrom, row.start, row.end))
        if(mat.shape[0] > dimension): #Make sure data is fetch right
            mat = mat[:dimension,:dimension]
        elif (mat.shape[0]<dimension):
            distance = (dimension + 1 - mat.shape[0])//2

            mat = mat[:dimension-(2*distance),:dimension-(2*distance)]

            mat = np.pad(mat, distance, mode='edge')

        

        yield mat
