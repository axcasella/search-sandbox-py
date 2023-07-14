import pytesseract
import glob, fitz

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

path = 'documents/'
all_files = glob.glob(path + "*.pdf")

for filename in all_files:
    print(filename)
    doc = fitz.open(filename)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(matrix=mat)  # render page to an image
        outputname = "data/out/page-%i.png" % page.number
        pix.save(outputname)  # store image as a PNG
        result = pytesseract.image_to_string(outputname, lang='eng')
        print(result)
