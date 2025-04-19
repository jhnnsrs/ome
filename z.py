from bioio_bioformats.biofile import BioFile

# cause dependency buildup
img = BioFile("test.tiff")
print(img.ome_metadata)
