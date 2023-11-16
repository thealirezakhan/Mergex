from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import webbrowser
from imutils.perspective import four_point_transform
import pytesseract
import streamlit as st 
from reportlab.pdfgen import canvas
from PIL import Image

count = 0
scale = 0.5

"""
# MergeX 🚀

"""

try:
    with st.form(key='my_form_to_submit'):
        image_file = st.file_uploader("Upload your file here...", type=["pdf"])
        submit_button = st.form_submit_button(label='Submit')

    global pdffile

    if submit_button:
        def load_image(image_file):
            img = Image.open(image_file)
            return img   
                
        if image_file is not None:
            imgdir = "./thealirezakhan/Mergex"
            with open(image_file.name, "wb") as f:
                f.write(image_file.getbuffer())
                    
            st.success("File saved")    

        
    global pdffile
    pdffile = image_file.name
    # print(pdffile)
    
    images = convert_from_path(pdffile,800, poppler_path="C:/Program Files/poppler-23.11.0/Library/bin")
    
                
    for i in range(len(images)):
        images[i].save('Image/pages'+ str(i) +'.jpg')

    for i in range(2):
        image = "Image/pages"+ str(i)
        print(image) 
    
    # global image , threshold , image_copy
    image = cv2.imread('Image/pages0.jpg')
    image = cv2.resize(image, (800, 1050))
    image_copy = image.copy()
        
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)

    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 1)

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (2,2,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
    img = img*mask2[:,:,np.newaxis]



    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    dst = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # cv2.imshow('dst', dst)

    edged = cv2.Canny(dst, 5, 655)
    # cv2.imshow("edged",edged)
    _, threshold = cv2.threshold(gray, 800, 510, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", threshold)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray2",gray2)



    global document_contour
    document_contour = np.array([[0, 0], [800, 0], [800, 1050], [0, 1050]])  

    contours, _ = cv2.findContours(gray2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
    # biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area
    # return biggest

    newimg = cv2.drawContours(image_copy, [document_contour], -1, (0, 0, 225), 3)
    # cv2.imshow("contour",newimg)
        
    warped = four_point_transform(image, document_contour.reshape(4, 2))
    # cv2.imshow("Warped", cv2.resize(warped, (int( warped.shape[1]), int( warped.shape[0]))))
    cv2.imwrite("converted/image0.jpg",warped)
    
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    image = cv2.imread('Image/pages1.jpg')
    image = cv2.resize(image, (800, 1050))
    image_copy = image.copy()
        
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)

    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= 1)

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (2,2,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
    img = img*mask2[:,:,np.newaxis]



    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    dst = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # cv2.imshow('dst', dst)

    edged = cv2.Canny(dst, 5, 655)
    # cv2.imshow("edged",edged)
    _, threshold = cv2.threshold(gray, 800, 510, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", threshold)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray2",gray2)



    document_contour = np.array([[0, 0], [800, 0], [800, 1050], [0, 1050]])  

    contours, _ = cv2.findContours(gray2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
    # biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if  area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area
    # return biggest

    newimg = cv2.drawContours(image_copy, [document_contour], -1, (0, 0, 225), 3)
    # cv2.imshow("contour",newimg)
        
    warped = four_point_transform(image, document_contour.reshape(4, 2))
    # cv2.imshow("Warped", cv2.resize(warped, (int( warped.shape[1]), int( warped.shape[0]))))
    cv2.imwrite("converted/image1.jpg",warped)
        
        


    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    

    def images_to_pdf(image_paths, output_pdf):
        c = canvas.Canvas(output_pdf)
        width = 800
        height = 1050

        for image_path in image_paths:
            c.setPageSize((800, 1050))
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Calculate aspect ratio to maintain the original image's proportions
            aspect_ratio = img_width / img_height
            new_width = min(img_width, width)
            new_height = new_width / aspect_ratio

            # Center the image on the page
            x_offset = (width - new_width) / 2
            y_offset = (height - new_height) / 2

            # Add the image to the PDF
            c.drawInlineImage(image_path, 0, 0, width=800, height=1050)

            # Add a new page for the next image
            c.showPage()
            

        c.save()      
          
    # Example usage:
    image_paths = ["converted/image0.jpg", "converted/image1.jpg"]
    output_pdf = "combined_images.pdf"

    images_to_pdf(image_paths, output_pdf)

    webbrowser.open_new_tab("file:///C:/Users/Ali%20Reza%20Khan/Downloads/CamScanner-In-Python-master/mergerX/combined_images.pdf")


    
    
    
except:
    pass    
