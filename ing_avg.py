import re
import nltk
import pandas as pd
import csv
from nltk.tokenize import word_tokenize,sent_tokenize
def calculate(essay):
    sentences = nltk.sent_tokenize(essay)
    wordcounts =[]
    ing_count=0
    sent_count=0
    for sentence in sentences:
        print sentence
        list_ing =re.findall(r'\b(\w+ing)\b',sentence)
        print list_ing
        count =len(list_ing)
        ing_count=ing_count+count
        words = sentence.split(' ')
        wordcounts.append(len(words))
        sent_count=sent_count+1
    avg_len = sum(wordcounts)/sent_count
    print sum(wordcounts)
    print (sent_count)
    print avg_len
    return avg_len ,ing_count  

def main(input_data):


    file =input_data
    
 
    essays = pd.read_csv(file)
    download_dir = "" #where you want the file to be downloaded to 
    
    _csv = open(download_dir, "w")
    
    _csv_file = open(download_dir, "w") 
    # Headers for CSV
    headers = ['Essay','ing','common_length']
    writer = csv.DictWriter(_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, fieldnames=headers, extrasaction='ignore')
    writer.writeheader()
    # Counter variable to know essay index
    essay_counter = 1

    for index, row in essays.iterrows():
        s1=row['Essay']
        s1=s1.decode("utf-8", "ignore")
        s1=s1.encode('ascii', 'ignore')
        #results=get_results(s1)
        
        
        avg_len,count=calculate(s1)
        
        # generate_graph(results_all, view = 0)
        # generate_graph(results, view = 0)
        
        output = {'Essay': s1,'ing':str(count),'common_length':str(avg_len)}
        writer.writerow(output)
        essay_counter = essay_counter + 1
        print "essay number"
        print essay_counter
        
      

     #pos_unique,count,ex_there_length,s_adj_length,pdt_length,c_conj_length,c_adj_length,s_adv_length,words,characters=syntax_results(s1)
     # dic = {essay}
     # for key in dic.keys():
     #     s1 = key
     #     s2 = dic[key]
     #     row = "\"" +essay +  "\"" +  "," + "\""   +pos_unique  +  "\"" + "," +  "\"" +ex_there_length  +  "\"" + "\n"
     #     csv.write(row)
                



main('pathtocsv')