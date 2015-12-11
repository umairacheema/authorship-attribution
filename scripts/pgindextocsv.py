#!/usr/bin/env python
#Utility script to convert Gutenberg data index file to CSV
#Gutenberg data when downloaded from pgiso.pglaf.org comes with
#index file containing metadata about downloaded eBooks
#This utility script converts that metadata into CSV format
import sys
import codecs
import re
import os
import csv
from bs4 import BeautifulSoup

#Function : print_help
#Purpose : Function to display help message
def print_help():
    print "Usage :"+sys.argv[0]+" <path to index.htm file> + <base url>"
    print " Where index.htm file is the index file downloaded via pgiso.pglaf.org"
    print "  and base url is the path prefix to be appended to retrieve URL"


#Function : get_book_name
#Purpose : Retrieves book name from the raw text
def get_book_name(raw):
    if raw is not None:
        pattern = '[^a-zA-Z0-9_ ]'
        prog = re.compile(pattern)
        raw = prog.sub('', raw)
        return raw
    else:
        return "Unknown"

#Function : get_author_name
#Purpose : Retrieves author's first and last name from the raw text
def get_author_name(raw):
    if raw is not None:
        raw = raw.replace(';',',')
        pattern = '[^a-zA-Z, ]'
        prog = re.compile(pattern)
        raw = prog.sub('',raw)
        raw = raw.strip()
        names = raw.strip(',')
        names = names.split(',')
        if len(names)>1:
            return names[1]+ " " + names[0]
        elif len(names)==1:
            return names[0]
    else:
        return "Unknown"
#Function : get_modified_url
#Purpose  : If user provides custom base url add that to 
#           file name
def get_modified_url(original,custom_base):
    url_parts = original.split('/')
    return custom_base + '/' + url_parts[2] + '/' + url_parts[3]


#Function : get_book_records
#Purpose : Function to retrieve book record
def get_book_records(file,base_url):
    book_records = []
    url = ""
    author_name = ""
    book_name = ""
    try:
        fh_index_file = codecs.open(file,'r','utf-8')
        index_data = fh_index_file.read()
    except IOError as e:
        print "I/O Error".format(e.errno, e.strerror)
        sys.exit(2)
    soup = BeautifulSoup(index_data,'html.parser')
    for link in soup.find_all('a',href=True):
        #skip useless links
        if link['href'] == '' or link['href'].startswith('#'):
            continue
        url = link.string
        if base_url is not None:
            url = get_modified_url(url,base_url)   
        etext = link.find_previous_sibling('font',text='EText-No.').next_sibling
        book_name = get_book_name(link.find_previous_sibling('font',text='Title:').next_sibling)
        author_name=get_author_name(link.find_previous_sibling('font',text='Author:').next_sibling)
        book_records.append({'etext':etext,'author':author_name.strip().strip('of ').lower(),'book':book_name.strip().lower(),'url':url.strip()})
    return book_records

#Function : write_csv_file
#Purpose : Writes book records to csv file
def write_csv_file(book_records):
    if os.path.exists('pgindex.csv'):
        os.remove('pgindex.csv')
    with open('pgindex.csv', 'w') as csvfile:
        fieldnames = ['etext', 'author','book','url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in book_records:
            writer.writerow(record)


#Check input arguments
if (len(sys.argv) < 2):
    print_help()
    sys.exit(1)

base_url = None
if (len(sys.argv) == 3):
   base_url = sys.argv[2] 

book_records_ = get_book_records(sys.argv[1],base_url)
write_csv_file(book_records_)

