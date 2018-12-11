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
import urllib2

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def reformat_str(inputstr):
    returnStr = ""
    tempStrArr = re.findall(r'\S+', inputstr) # split input string to temporary string array, based on whitespace,tab and \n
    for tempStr in tempStrArr:
        if not is_integer(tempStr): # ignore integer strings
            returnStr += (tempStr.lower() + " ") # make it lowercase and append it to end of result string with 1-char whitespace delimiter
    return returnStr

def convert(fname, pages=None):
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

docs = []
options = Options()
options.set_headless(True) # do not open browser in gui
browser = webdriver.Firefox(options=options)
browser.get("http://www.semanticscholar.org/search?q=\"Ali Fuat Alkaya\"&sort=relevance&pdf=true")
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
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r'alkaya')
        if not os.path.exists(final_directory): # create dir if it is not exist
            os.makedirs(final_directory)

            for link in links:
                if link.startswith('https://pdfs') == True: # it is the link what we want(pdf's link) !!!
                    final_pdf_path = os.path.join(final_directory, str(count) + ".pdf") # it will download to '.../Alkaya/1.pdf'
                    filedata = urllib2.urlopen(link) # Download it!
                    datatowrite = filedata.read()
                    with open(final_pdf_path, 'wb') as pdfFile:  
                        pdfFile.write(datatowrite)
                    # download finished, extract text from it and save it to the docs array
                    docs.append(convert(final_pdf_path))
                    count += 1



tfidf = TfidfVectorizer()

response = tfidf.fit_transform(docs)

feature_names = tfidf.get_feature_names()
for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[0, col])
