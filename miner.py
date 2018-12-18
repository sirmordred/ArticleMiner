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
import csv
from wordcloud import WordCloud # pip install wordcloud, pip install matplotlib, apt-get install python-tk
import numpy as np
from PIL import Image

STOP_WORDS = [ # taken from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py and added extra scientific stop words like 'introduction', 'university'
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", 
    "abstract", "introduction", "conclusions", "related", "work", "author", "university", # extra stop_words row
    "solution", "solutions", "problem", "problems"] # extra stop_words row

def only_contain_letters(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False

def pre_process_document(inputstr):
    returnStr = ""
    tempInputStr = inputstr.replace('"', ' ') # there may be some quoted words too
    tempStrArr = re.findall(r'\S+', tempInputStr) # split input string to temporary string array, based on whitespace,tab and \n
    for tempStr in tempStrArr:
        if len(tempStr) > 1 and only_contain_letters(tempStr): # ignore chars(1-char strings) and non-alphabetic strings
            tempStrLow = tempStr.lower() # make it lowercase before searching it in stop_words_list
            if tempStrLow not in STOP_WORDS:
                returnStr += (tempStrLow + " ") # if its not in stop_words_list, append it to end of result string with 1-char whitespace delimiter
    return returnStr

def pdf_to_document(fname):
    pagenums = set()
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
    return text

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

def contain_documents(author):
    pdf_directory = os.path.join(os.getcwd(), author)
    if os.path.exists(pdf_directory): # check existence
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".txt"):
                return True
    return False

def articles_to_documents(author): # returns list of documents (to send tfidvectorizer)
    cnt = 1
    pdf_directory = os.path.join(os.getcwd(), author)
    if os.path.exists(pdf_directory) and not contain_documents(author): # only run if pdf files are exist and txt files are not
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                dirtyDoc = pdf_to_document(os.path.join(pdf_directory, filename)) # extract text from pdf
                cleanDoc = pre_process_document(dirtyDoc) # strip/remove non-alphabetic chars like digits, quotes and also remove stop_words
                final_txt_path = os.path.join(pdf_directory, str(cnt) + ".txt")
                with open(final_txt_path, 'wb') as txtFile:  
                                    txtFile.write(cleanDoc) # write cleaned document to txt
                cnt += 1

def get_documents(author):
    docs = []
    txt_directory = os.path.join(os.getcwd(), author)
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            final_txt_path = os.path.join(txt_directory, filename)
            with open(final_txt_path, 'r') as txtFile:
                doc=txtFile.read().replace('\n', '')
            docs.append(doc) # append document to docs array
    return docs

def write_to_csv(output_list, is_tf):
    outputFile = 'defaultOutputValues.csv'
    if is_tf:
        outputFile = 'tf_list.csv'
    else:
        outputFile = 'tfidf_list.csv'
    with open(outputFile, mode='w') as csv_output_file:
        csv_writer = csv.writer(csv_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row_output_list in output_list:
            row_array = row_output_list.split(',')
            word = row_array[0]
            word_value = row_array[1]
            csv_writer.writerow([word, word_value])

def generate_wordcloud(author, doclist):
    data = ""
    for docElem in doclist:
        data += docElem
    marmara_mask = np.array(Image.open("mu.png"))
    wc = WordCloud(collocations=False,
                    background_color ='white',
                    mask=marmara_mask,
                    max_words=2000).generate(data)
    wc.to_file(author+"_wordcloud.png")

download_articles('Ali Fuat Alkaya')
articles_to_documents('Ali Fuat Alkaya')
corpus = get_documents('Ali Fuat Alkaya')

generate_wordcloud('Ali Fuat Alkaya',corpus)

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
