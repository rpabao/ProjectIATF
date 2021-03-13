# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:37:50 2021

@author: Roland P. Abao
"""
# Import libraries 
import re
import cv2
import glob
import pytesseract 
import pandas as pd
from PIL import Image 
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from pdf2image import convert_from_path 

from enchant.checker import SpellChecker
special_words = ['coronavirus','covid','ncov','cov','wuhan','hubei','covax',
                 'biontech','lgus','lgu','comorbidities','Listahanan','sars',
                 'dfa','dilg','doj','dotr','doh','pcr','roa','duterte']


english_spell_checker = SpellChecker("en_US")
for sp in special_words: english_spell_checker.add(sp)

def is_in_english(quote, max_error_count = 3, min_text_length = 3):
  english_spell_checker.set_text(quote)
  errors = [err.word for err in english_spell_checker]
  errors = list(set(errors)) 
  return False if ((len(errors) > max_error_count) or 
                   len(quote.split()) < min_text_length) else True

def remove_header_from_image(imageFilePath, threshold=0.06):
    template_files = [f for f in glob.glob("images/header_templates/*.jpg")]
    
    for headerFilePath in template_files:
        # Read the images from the file
        header_image = cv2.imread(headerFilePath)
        orig_image = cv2.imread(imageFilePath)
        
        # Match the tempplate to the original image
        result = cv2.matchTemplate(header_image, orig_image, cv2.TM_SQDIFF_NORMED)
    
        # We want the minimum squared difference
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)
        
        
        if mn < threshold:
            # Draw the rectangle:
            # Extract the coordinates of our best match
            MPx,MPy = mnLoc
            
            # Step 2: Get the size of the template. This is the same size as the match.
            h_rows,h_cols = header_image.shape[:2]
            o_rows,o_cols = orig_image.shape[:2]
            
            if (MPx+h_cols) <= o_rows and (MPy+h_rows) <= o_cols:
                # Step 3: Draw the rectangle on orig_image
                cv2.rectangle(orig_image, (MPx,MPy),(MPx+h_cols,MPy+h_rows),(255,255,255),-1)
                # Save the reulting image
                cv2.imwrite(imageFilePath,orig_image)
                print('.', end ='')
                break

def remove_signature_from_image(imageFilePath, threshold=0.06):
    template_files = [f for f in glob.glob("images/signature_templates/*.jpg")]
    
    for headerFilePath in template_files:
        # Read the images from the file
        header_image = cv2.imread(headerFilePath)
        orig_image = cv2.imread(imageFilePath)
        
        # Match the tempplate to the original image
        result = cv2.matchTemplate(header_image, orig_image, cv2.TM_SQDIFF_NORMED)
    
        # We want the minimum squared difference
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)
        ###print(str(mn))
        
        if mn < threshold:
            # Draw the rectangle:
            # Extract the coordinates of our best match
            MPx,MPy = mnLoc
            
            # Step 2: Get the size of the template. This is the same size as the match.
            h_rows,h_cols = header_image.shape[:2]
            o_rows,o_cols = orig_image.shape[:2]
            
            # Step 3: Draw the rectangle on orig_image
            cv2.rectangle(orig_image, (MPx,MPy),(MPx+h_cols,MPy+h_rows),(255,255,255),-1)
            # Save the reulting image
            cv2.imwrite(imageFilePath,orig_image)
            print('.', end ='')


def textMinePDF(filepath, dpi=200, thread_count=4, bottom_margin=14/15, n_ngram=3,
                body_cue='whereas', footer_cue=['approved','this is to certify that']):
    
    ''' 
    Part #1 : Converting PDF to images 
    '''
    # Store all the pages of the PDF in a variable 
    pages = convert_from_path(filepath,dpi=dpi,thread_count=thread_count) 
    print('.', end ='')
    
    # Counter to store images of each page of PDF to image 
    image_counter = 1
    
    # Iterate through all the pages stored above 
    for page in pages: 
    
    	# Declaring filename for each page of PDF as JPG 
    	filename = "images/page_"+str(image_counter)+".jpg"
    	
    	# Save the image of the page in system 
    	page.save(filename, 'JPEG') 
        
        # Remove header from the image
    	remove_header_from_image(filename)
        
        # Remove signature if found
    	remove_signature_from_image(filename)
    
    	# Increment the counter to update filename 
    	image_counter = image_counter + 1
        
    
    ''' 
    Part #2 - Recognizing text from the images using OCR 
    '''
    
    # Variable to get count of total number of pages 
    filelimit = image_counter-1
    
    text = ''
    
    # Iterate from 1 to total number of pages 
    for i in range(1, filelimit + 1): 
    
    	# Set filename to recognize text from 
    	filename = "images/page_"+str(i)+".jpg"
    
        # crop image a little bit at the top and bottom
    	image = Image.open(filename)
    	width, height = image.size   # Get dimensions
    	left = 0
    	top = 0
    	right = width
    	bottom = bottom_margin * height
    	cropped_image = image.crop((left, top, right, bottom))
    	cropped_image.save(filename)
    	print('.', end ='')
    		
    	# Recognize the text as string in image using pytesserct 
    	text = text + '\n\n' + str(((pytesseract.image_to_string(cropped_image)))) 
    	print('.', end ='')   
    
    
    ''' 
    Part #3 - Preprocessing of text
    '''
    
    # The recognized text is stored in variable text 
    # Any string processing may be applied on text 
    # Here, basic formatting has been done: 
    # In many PDFs, at line ending, if a word can't 
    # be written fully, a 'hyphen' is added. 
    # The rest of the word is written in the next line 
    # Eg: This is a sample text this word here GeeksF- 
    # orGeeks is half on first line, remaining on next. 
    # To remove this, we replace every '-\n' to ''. 
    text = text.replace('-\n', '')	 
    
    # Remove all symbols
    text = re.sub('[^\w\n]', ' ', text)
    
    # Convert text to lower case
    text_lower = text.lower()
    
    # Get the header and the rest of the document
    header, body = text_lower.split(body_cue,1)
    body = body_cue + body
    
    # Get the body and footer of the document
    footer = ''
    for cue in footer_cue:
        body = body.split(cue)
        if len(body)>1:
            footer = cue + body.pop() + footer
        body = cue.join(body)
    
    # Find the resolution number
    try:
        resolution = int(header.split('resolution no')[1].split('\n')[0])
    except:
        resolution = float("NaN")
    
    # Find the series year
    try:
        series = int(header.split('series of')[1].split('\n')[0])
    except:
        series = float("NaN")
    
    
    ''' 
    Part #4 - Processing of text per paragraphs
    '''
    
    # Split body by paragraph and remove non-english sentences
    paragraphs_raw = [p.strip() for p in body.split('\n\n')]
    english_paragraphs = []
    for p_r in paragraphs_raw:
        p_r_unit = ''
        for line in p_r.split('\n'):
            if is_in_english(line,min_text_length=2):
                p_r_unit += line + ' '
        if p_r_unit != '':
            p_r_unit = ' '.join([word for word in p_r_unit.split() if len(word)>1])
            english_paragraphs.append(p_r_unit)
    english_paragraphs = [p for p in english_paragraphs if is_in_english(p,max_error_count=5)]
    
    print('.', end ='')
    
    
    ''' 
    Part #5 - Processing of text per ngrams
    '''
    
    # remove stop words and words with less than 3 characters
    stop_words = set(stopwords.words('english'))
    stop_words.update(['whereas','shall','via'])
    body_list = [w for w in word_tokenize(body) if len(w)>2 and (not w in stop_words)]
    
    # process ngrams
    ngram_result = [' '.join(ng) for ng in ngrams(body_list, n_ngram)]
    print('.', end ='')
    
    
    ''' 
    Part #6 - Create and return the dataframes
    '''
    # return resulting dataframe
    return (pd.DataFrame(list(zip([resolution for i in range(len(english_paragraphs))], 
                           [series for i in range(len(english_paragraphs))],
                           english_paragraphs)), columns =['resolution', 'series','item']),
            pd.DataFrame(list(zip([resolution for i in range(len(ngram_result))], 
                           [series for i in range(len(ngram_result))],
                           ngram_result)), columns =['resolution', 'series',str(n_ngram)+'-gram']))



# Find all the pdf files in the input folder
PDF_files = [f for f in glob.glob("input/*.pdf")]

result_df1 = pd.DataFrame() # per paragraph
result_df2 = pd.DataFrame() # per n-gram
counter = 1
total_count = str(len(PDF_files))

# Text mine each pdf file in the input folder
for file in PDF_files:
    print('\nProcessing ' + str(counter) + ' out of ' + total_count, end =' ')
    (temp_df1,temp_df2) = textMinePDF(file)
    result_df1 = pd.concat([result_df1,temp_df1])
    result_df2 = pd.concat([result_df2,temp_df2])
    counter += 1

# Save to csv
outputFilePath1 = 'output/output_per_paragraph.csv'
outputFilePath2 = 'output/output_per_ngram.csv'
result_df1.to_csv(outputFilePath1,index=False)
result_df2.to_csv(outputFilePath2,index=False)
print('\n\nResults generated at ' + outputFilePath1 + ' and ' + outputFilePath2)
