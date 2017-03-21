## THIS CODE WAS DEVELOPED FROM A TEMPLATE IN DR. MATTHEW BERLAND'S
## COMPUTATIONAL RESEARCH METHODS COURSE AT UNIVERSITY OF WISCONSIN-MADISON

## if packages are not installed, go to Terminal and enter "conda install [name of package]"
import csv
import nltk
from nltk.collocations import *
from nltk.corpus import stopwords
import gensim
import pickle
from gensim.models import Word2Vec
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

max_lines = 1000 ##this is set in case you have gigantic csv that might crash

## this removes words that are not helpful for analysis, like "the"
whitelist = pickle.load(open("lowercase_words.pkl"))
stoplist = stopwords.words('english')

sentence_stream = [] ## a list for individual student responses in notebooks
student_stream = [] ## a list for the individual student ID
group_stream = [] ## a list for group ID

## this section opens the csv, puts it in a dictionary for easy reading, and then tokenizes words
## in the next line, change the path/name to the path/name of your csv
## it helps to put your csv in the same folder as lowercase_words.pkl, vader_lexicon.txt, etc.
with open("/Sample_CSV.csv", "rU") as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        if max_lines < 0:
            break
        if max_lines >= 0:
            max_lines -= 1
            # print line["p19_1"] ## check individual lines for bad unicode symbols - will need to fix these in the csv
            question_line = str(line["p19_1"]) ## change p19_1 to the header of the column that you want to analyze
            group_ID = str(line["Group"]) ## making lists - change these to fit your needs
            stu_ID = str(line["Student_Code"])
            question_line = question_line.encode('ascii', 'ignore').lower() ## where the magic starts
            question_words = nltk.wordpunct_tokenize(question_line)
            ## many short words are not useful for analysis
            ## you can change len(word) > [number] below to reflect what you need (length = number of characters)
            question_words = [word for word in question_words if (len(word) > 3 and word not in stoplist and word in whitelist)]
            sentence_stream.append(question_words)
            student_stream.append(stu_ID) ## these should match the lists you created above
            group_stream.append(group_ID)

## this makes the data more "readable" by flattening it into a list
def shallow_flatten(shallow_list):
    return [item for sublist in shallow_list for item in sublist]
all_ordered_words = shallow_flatten(sentence_stream)

## this will return words that frequently occur in the same position as the similarity words you enter below
model = Word2Vec(sentence_stream)
similarity_words = ["factors", "ecosystem","depend","interact"] ## adjust these according to your theory/data
for similarity_word in similarity_words:
    if similarity_word in model:
        print similarity_word.rjust(10),"==>",' '.join(map(lambda x : x[0] , model.most_similar(positive=[similarity_word])[:4]))

## this section is where the tf-idf algorithm comes in
## you can use other gensim algorithms too, like LSA
bigram_modeler = gensim.models.Phrases(sentence_stream, threshold=3)
bigrammed_sentence_stream = bigram_modeler[sentence_stream]
dictionary = gensim.corpora.Dictionary(bigrammed_sentence_stream)
raw_corpus = [dictionary.doc2bow(t) for t in bigrammed_sentence_stream]
tfidf = gensim.models.TfidfModel(raw_corpus) ## this is where you change algorithms, if desired
sid = SentimentIntensityAnalyzer() ## this is VADER sentiment analysis, which work really well for social media
#intensity_threshold = 0.001 ## adjustable - I scrapped this section because I wasn't focusing on sentiment
for sentence_index in range(len(bigrammed_sentence_stream)):
    sentence = bigrammed_sentence_stream[sentence_index]  ## notice these lists are repeating again
    stu_ID = student_stream[sentence_index]
    group_ID = group_stream[sentence_index]
    sentiment = sid.polarity_scores(" ".join(sentence)) ##VADER assigns positive, negative, and neutral scores
    positivity = sentiment["pos"]
    negativity = sentiment["neg"]
    neutrality = sentiment["neu"]
    intensity = sentiment["compound"] ## this is the intensity of the sentiment
    combine = float(positivity) - float(negativity) ## this subtracts negative score from positive to get a net positive score
    if len(sentence) > 3: # and (positivity > intensity_threshold or negativity > intensity_threshold):
        sentence_tf = tfidf[dictionary.doc2bow(sentence)]
        remapped_sentence = map(lambda x: (dictionary[x[0]],x[1]), sentence_tf) # [("bob",0.5),...]
        print stu_ID + "," + group_ID ## prints individual student ID and group ID
        print "[+{:3d}, -{:3d}, {:3d}, {:3d}]".format(int(100*positivity),int(100*negativity),int(100*neutrality),int(100*intensity)), ## polarity scores
        print " ".join(map(lambda y: y[0],sorted(remapped_sentence,reverse=True,key=lambda x:x[1])[0:4])) ## prints "unique" words for each entry

## finds bigrams (two-word sequences) - may or may not be helpful
finder = BigramCollocationFinder.from_words(all_ordered_words)
finder.apply_freq_filter(min(50,max(10,max_lines/100))) # 10 < (lines/100) < 50
bigram_measures = nltk.collocations.BigramAssocMeasures()
ten_best = finder.nbest(bigram_measures.pmi,10)
for a_bigram in ten_best:
    print "_".join(a_bigram),
print ""
