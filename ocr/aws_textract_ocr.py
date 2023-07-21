import boto3
from PIL import Image, ImageDraw
import glob, fitz
from textractor import Textractor
from textractor.data.constants import TextractFeatures
import pandas as pd

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

path = '../documents/'
all_files = glob.glob(path + "*.pdf")

# AWS client
client = boto3.client('textract', region_name='us-east-1', aws_access_key_id='AKIA35CIUIWYY67GAQE5', aws_secret_access_key='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')
extractor = Textractor(profile_name="default")

for filename in all_files:
    print(filename)
    doc = fitz.open(filename)  # open document
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap(matrix=mat)  # render page to an image
        outputname = "../data/out/page-%i.png" % page.number
        pix.save(outputname)  # store image as a PNG

        with open(outputname, 'rb') as image:
            img_bytes = bytearray(image.read())
            
        # If not using textractor
        # response = client.analyze_document(
        #     Document={'Bytes': img_bytes},
        #     FeatureTypes=['FORMS'])
        # print(response)

        document = extractor.analyze_document(
        file_source=outputname,
        features=[TextractFeatures.FORMS],
)
        # Getting list of keys and list of values
        keys = [kv.key.text for kv in document.key_values]
        values = [str(kv.value) for kv in document.key_values]

        # Creating the data frame with the values and assigning the keys as columns
        df = pd.DataFrame([values], columns=keys)

        filename = "output.csv"

        # Write the dataframe object into CSV file
        df.to_csv(filename, index=None, header=True)

        # To verify if the CSV is created, read CSV file and convert it into dataframe object
        df = pd.DataFrame(pd.read_csv("output.csv"))

        print(df)





