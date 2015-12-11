#!/usr/bin/env python
import sys
import re
import os
import csv
from pymarc import MARCReader

#Language of interest
LANGUAGE = "eng"
#Constants for MARC File format
LANGUAGE_RECORD_FIELD = '008'
URI_RECORD_FIELD = '856'
LANGUAGE_CODE_START_INDEX = 41
LANGUAGE_CODE_END_INDEX = 44


#Function : clean_metadata
#Purpose : Function to clean metadata records
#          Removes special characeters
def clean_metadata(raw):
    if raw is not None:
        pattern = '[^a-zA-Z0-9 ]'
        prog = re.compile(pattern)
        cleaned = prog.sub('', raw)
        return cleaned
 

#Function : get_metadata
#Purpose : Function to retrieve metadata from MARC record
def get_metadata(record):
    #Get language :MARC Code 008        
    language_record = str(record[LANGUAGE_RECORD_FIELD])
    if language_record is not None:
        if len(language_record) > LANGUAGE_CODE_END_INDEX:
            language_code = language_record[LANGUAGE_CODE_START_INDEX:LANGUAGE_CODE_END_INDEX]
            #Only proceed if language is language of interest
            if(language_code == LANGUAGE):
                #Find URI to access file: MARC Code 856
                url = str(record[URI_RECORD_FIELD]['u'])
                title = record.title()
                if title is None:
                    title = "Unknown"
                author = record.author()
                if author is None:
                    author = "Unknown"
                title = clean_metadata(title.encode('utf-8'))
                author = clean_metadata(author.encode('utf-8'))
                return (title,author,url)

#Function : get_etext_number
#Purpose : Given metadata, retrieves the eText number of the book
def get_etext_number(metadata):
    if metadata is not None:
        url = metadata[2]
        filename = url[url.rindex('/')+1:]
        return filename 
 

etexts = []
#Remove previous marcs.csv if it exists
if os.path.exists('marcs.csv'):
    os.remove('marcs.csv')
with open('marcs.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile)
    with open('../data/catalog.marc','r') as fh:
        reader = MARCReader(fh)
        for record in reader:
            #Get metadata for the book
            metadata = get_metadata(record)
            if metadata is not None:
                filewriter.writerow([metadata[1],metadata[0],metadata[2]])
            etext = get_etext_number(metadata)
            if etext is not None:
                etexts.append(int(etext))       

print "Minimum eText is:"+ str(min(etexts))
print "Mazimum eText is:"+ str(max(etexts))
