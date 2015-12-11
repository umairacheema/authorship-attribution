#!/usr/bin/env python
# Creates Corpus from pgindex.csv created by pgindextocsv.py
# Requires following python libraries installed on the system
# gutenberg,boto
# Put S3 bucket credentials in /etc/boto.cfg before
# running this script
# The script will read books by each author from S3 and 
# create a single document containing all books
# concatenated by a single author
# The gutenberg headers and footers are removed
# Moreover any left over noise after removing the header/footer
# is further removed by removing top 5% and bottom 5% of the book contents. 

from gutenberg.cleanup import strip_headers
from boto.s3.key import Key
import pandas as pd
import boto
import re

#Set Pandas options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

S3_BUCKET_NAME = 'cs109-gutenberg'

#Connect with S3
s3 = boto.connect_s3()
bucket = s3.get_bucket(S3_BUCKET_NAME)

#Create data frame from the csv file
df = pd.read_csv('../data/pgindex.csv')

for index, record in df.iterrows():
    # Get the key
    url = record['url']
    key = bucket.get_key(url)
    if key is None:
        #Remove the utf8 extension from url
        utf8_extension = url.rfind('.utf8')
        if (utf8_extension != -1):
            url = url[0:utf8_extension]
    key = bucket.get_key(url)
    if key is None:
        continue
    contents = key.get_contents_as_string()
    contents = unicode(contents, 'utf-8')
    book_text = strip_headers(contents).strip()
    book_length = len(book_text)
    noise_size = int(book_length * 0.05) 
    #Compute offsets for content
    start_offset = noise_size
    end_offset = book_length - noise_size
    #Remove the noise from book text
    document = book_text[start_offset:end_offset]
    #Truncate the document at full stops
    start = document.find('.')
    end = document.rfind('.')
    if ((start != -1) and (end != -1)):
        document = document[start+1:end+1]
    #Remove special characters and digits
    pattern = '[^\w+.\s+,:;?\'-]'
    prog = re.compile(pattern,re.UNICODE)
    document = prog.sub('',document)
    document = re.sub(" \d+ |\d+.",'', document)
    #Check if this author exists in the corpus
    author_name = str(record['author']).replace(' ','_')
    key = bucket.get_key('corpus/'+author_name+'.txt')
    if key is None:
        #Create the key
        k = Key(bucket)
        k.key = 'corpus/'+author_name+'.txt'
        k.set_contents_from_string(document)
    else:
        previous_contents = key.get_contents_as_string()
        previous_contents = unicode(previous_contents, 'utf-8')
        updated_document = previous_contents + document
        key.set_contents_from_string(updated_document)




