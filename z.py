from bioio import BioImage

# cause dependency buildup
img = BioImage("test.tiff")
print(img.ome_metadata)
