import PyPDF2
from PIL import Image

import warnings
from os import path



warnings.filterwarnings("ignore")


def new_extract_images_from_pdf(filename, num_pages, dest_dir):
    number = 0

    def recurse(page, xObject):
        global number

        xObject = xObject['/Resources']['/XObject'].getObject()

        for obj in xObject:

            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()

                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    # todo - currently manually set to RGB
                    mode = "RGB"

                imagename = "%s - p. %s"% (obj[1:], p)

                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    img.save(dest_dir + imagename + ".png")
                    number += 1

                # todo
                # elif xObject[obj]['/Filter'] == '/DCTDecode':
                #     img = open(imagename + ".jpg", "wb")
                #     img.write(data)
                #     img.close()
                #     number += 1
                # elif xObject[obj]['/Filter'] == '/JPXDecode':
                #     img = open(imagename + ".jp2", "wb")
                #     img.write(data)
                #     img.close()
                #     number += 1
            else:
                recurse(page, xObject[obj])

    abspath = path.abspath(filename)
    pdf_file = PyPDF2.PdfFileReader(open(filename, "rb"))

    for p in range(num_pages):
        page0 = pdf_file.getPage(p - 1)
        recurse(p, page0)

    print('%s extracted images' % number)


def extract_images_from_pdf(filename):
    # https://nedbatchelder.com/blog/200712/extracting_jpgs_from_pdfs.html
    pdf = open(filename, "rb").read()

    startmark = "\xff\xd8"
    startfix = 0
    endmark = "\xff\xd9"
    endfix = 2
    i = 0

    njpg = 0
    while True:
        istream = pdf.find("stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream + 20)
        if istart < 0:
            i = istream + 20
            continue
        iend = pdf.find("endstream", istart)
        if iend < 0:
            raise Exception("Didn't find end of stream!")
        iend = pdf.find(endmark, iend - 20)
        if iend < 0:
            raise Exception("Didn't find end of JPG!")

        istart += startfix
        iend += endfix
        print("JPG %d from %d to %d" % (njpg, istart, iend))

        jpg = pdf[istart:iend]
        jpgfile = open("jpg%d.jpg" % njpg, "wb")
        jpgfile.write(jpg)
        jpgfile.close()

        njpg += 1
        i = iend