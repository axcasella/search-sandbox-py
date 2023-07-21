import easyocr
import glob, fitz
import pandas as pd

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

path = '../documents/'
all_files = glob.glob(path + "*.pdf")

reader = easyocr.Reader(['en'], gpu=False)

for filename in all_files:
    # print(filename)
    doc = fitz.open(filename)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(matrix=mat)  # render page to an image
        outputname = "../data/out/page-%i.png" % page.number
        pix.save(outputname)  # store image as a PNG
        result = reader.readtext(outputname)

        dataframe = pd.DataFrame(result, columns = ['bound_box', 'text', 'confidence'])
        pd.set_option('display.max_rows', None)
        print(dataframe)



