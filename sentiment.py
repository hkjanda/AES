from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    neg =score["neg"]
    pos=score["pos"]
    neu=score["neu"]
    compound=score["compound"]
    return neg,pos,neu,compound

def main(input_data):


    file =input_data
    
 
    essays = pd.read_csv(file)
    download_dir = "/home/compute/work/aee/essay_evaluation_codes/domain4/sentiment.csv" #where you want the file to be downloaded to 
    
   
    
    _csv_file = open(download_dir, "w") 
    # Headers for CSV
    headers = ['Essay','neg','pos','neu','compound']
    writer = csv.DictWriter(_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, fieldnames=headers, extrasaction='ignore')
    writer.writeheader()
    # Counter variable to know essay index
    essay_counter = 1

    for index, row in essays.iterrows():
        s1=row['Essay']
    
        #results=get_results(s1)
        
        
        neg,pos,neu,compound=sentiment_analyzer_scores(s1)
   
        
        output = {'Essay': s1,'neg':str(neg),'pos':str(pos),'neu':str(neu),'compound':str(compound)}
        writer.writerow(output)
        # essay_counter = essay_counter + 1
        # print "essay number"
        # print essay_counter
        
      

     #pos_unique,count,ex_there_length,s_adj_length,pdt_length,c_conj_length,c_adj_length,s_adv_length,words,characters=syntax_results(s1)
     # dic = {essay}
     # for key in dic.keys():
     #     s1 = key
     #     s2 = dic[key]
     #     row = "\"" +essay +  "\"" +  "," + "\""   +pos_unique  +  "\"" + "," +  "\"" +ex_there_length  +  "\"" + "\n"
     #     csv.write(row)
                



main('/home/compute/work/aee/essay_evaluation_codes/domain4/ALL.csv')