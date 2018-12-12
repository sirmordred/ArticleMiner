from selenium import webdriver  # pip install selenium NOTE: also geckodriver must be in PATH
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options
from lxml import html
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter  # pip install pdfminer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import TfidfVectorizer  # pip install scikit-learn
import os
import re
import urllib2

def only_contain_letters(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False

def reformat_str(inputstr):
    returnStr = ""
    tempStrArr = re.findall(r'\S+', inputstr) # split input string to temporary string array, based on whitespace,tab and \n
    for tempStr in tempStrArr:
        if only_contain_letters(tempStr): # ignore non-alphabetic strings
            returnStr += (tempStr.lower() + " ") # make it lowercase and append it to end of result string with 1-char whitespace delimiter
    return returnStr

def get_text_from_pdf(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums): # iterate for the next pages of pdf
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return reformat_str(text) # reformat string before returning

def func(instr):
    return float(instr.split('-')[1])

def download_articles(author):
    pdf_dir = os.path.join(os.getcwd(), author) # get path of pdf dir
    if not os.path.exists(pdf_dir): # check existence
        options = Options()
        options.set_headless(True) # do not open browser in gui
        browser = webdriver.Firefox(options=options)
        browser.get("http://www.semanticscholar.org/search?q=\"" + author + "\"&sort=relevance&pdf=true")
        count = 1
        delay = 10 # should be enough
        try:
            myElem = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'result-count')))
        except TimeoutException:
            print ("Loading took too much time!")
        pageSource = browser.page_source
        if pageSource != "": # if result page source is not empty
            htmlRet = html.fromstring(pageSource) # convert it to html object
            links = htmlRet.xpath('//a/@href') # get all urls which are in href attributes of html object
            if len(links) > 0:
                os.makedirs(pdf_dir) # create directory which will hold all article pdfs
                for link in links:
                    if link.startswith('https://pdfs') == True: # it is the link what we want(pdf's link) !!!
                        final_pdf_path = os.path.join(pdf_dir, str(count) + ".pdf") # it will download to '.../Alkaya/1.pdf'
                        filedata = urllib2.urlopen(link) # Download it!
                        datatowrite = filedata.read()
                        with open(final_pdf_path, 'wb') as pdfFile:  
                            pdfFile.write(datatowrite)
                        count += 1

def articles_to_documents(author): # returns list of documents (to send tfidvectorizer)
    docs = []
    pdf_directory = os.path.join(os.getcwd(), author)
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            docs.append(get_text_from_pdf(os.path.join(pdf_directory, filename)))
    return docs

download_articles('Ali Fuat Alkaya')
corpus = articles_to_documents('Ali Fuat Alkaya')

tfidf = TfidfVectorizer()

response = tfidf.fit_transform(corpus)

feature_names = tfidf.get_feature_names()
unsortedList = []
for col in response.nonzero()[1]:
#    print('{0: <20} {1}'.format(feature_names[col],response[0, col]))
    unsortedList.append(feature_names[col] + '-' + str(response[0, col]))

sortedList = sorted(unsortedList,key=func,reverse=True)
for outstr in sortedList:
    print(outstr)
