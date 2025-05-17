from prepro import *
from preprocessing import *
import pandas as pd
from difflib import SequenceMatcher
from tqdm import tqdm
# pharm lexicon

    
stikshh = [
    ".",
    " ",
    "-",
    "_",
    "+",
    "w",
    "°",
    "?",
    ";",
    "!",
    ":",
    "(",
    ")",
]  # unwanted chars
stiksh = [
    ".",
    " ",
    "-",
    "_",
    "+",
    "w",
    "°",
    "?",
    ";",
    "!",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]  # unwanted chars that may repeat

with open(
    "lexicon/EmotionLookupTable.txt", "r", encoding="utf-8"
) as file:
    terms_list = file.read().splitlines()

word = []  # 2 arrays for word and score
score = []

for t in terms_list:
    t = t.split("	")
    word.append(t[0])
    score.append(t[1])

for i in range(0, len(score)):
    score[i] = int(score[i])  # make int from string

for i in range(0, len(word)):
    word[i] = clean_accent(word[i].lower())  # clean accent of word


# emoticontable same as pharm
with open(
    "lexicon/EmoticonLookupTable.txt", "r", encoding="utf-8"
) as file:
    emotic_list = file.read().splitlines()
emot = []
scorem = []
for te in emotic_list:
    te = te.split("	")
    emot.append(te[0])
    scorem.append(te[1])
for i in range(0, len(scorem)):
    scorem[i] = int(scorem[i])


# boosterwords same as before
with open("lexicon/BoosterWordList.txt", "r", encoding="utf-8") as file:
    terms_listbo = file.read().splitlines()

boost = []
scorebo = []

for tb in terms_listbo:
    tb = tb.split("	")
    boost.append(tb[0])
    scorebo.append(tb[1])
for i in range(0, len(scorebo)):
    scorebo[i] = int(scorebo[i])
for i in range(0, len(boost)):
    boost[i] = clean_accent(boost[i].lower())

# negwords
with open("lexicon/NegatingWordList.txt", "r", encoding="utf-8") as file:
    terms_listneg = file.read().splitlines()
neg = []
for tn in terms_listneg:
    tn = tn.split("	")
    neg.append(tn[0])
for i in range(0, len(neg)):
    neg[i] = clean_accent(neg[i].lower())


def clean_accent(text):

    t = text

    # el
    t = t.replace("Ά", "Α")
    t = t.replace("Έ", "Ε")
    t = t.replace("Ί", "Ι")
    t = t.replace("Ή", "Η")
    t = t.replace("Ύ", "Υ")
    t = t.replace("Ό", "Ο")
    t = t.replace("Ώ", "Ω")
    t = t.replace("ά", "α")
    t = t.replace("έ", "ε")
    t = t.replace("ί", "ι")
    t = t.replace("ή", "η")
    t = t.replace("ύ", "υ")
    t = t.replace("ό", "ο")
    t = t.replace("ώ", "ω")
    t = t.replace("ς", "σ")
    t = t.replace("♡", "")
    t = t.replace("☆", "")
    t = t.replace("*", "")

    return t
def spell_check(text):
    corrected_text = []
    for word in text.split():
        if hspell.spell(word):
            corrected_text.append(word)
        else:
            suggestions = hspell.suggest(word)
            corrected_text.append(suggestions[0] if suggestions else word)
    return " ".join(corrected_text)
with open("datasets/stopwords_greek.csv", encoding="utf-8") as f:
    stopwords = set(line.strip() for line in f if line.strip())
    
df=pd.read_csv('rawreviews.csv')

print(f'len df before {len(df)}')
df=df.dropna(subset=['agg_rating','comment'])
print(f'len df after drop nan agg rating and comment {len(df)}')
# retrieve all aspect names from the dataset
aspect_list = ['Ποιότητα κλήσης', 'Φωτογραφίες', 'Καταγραφή Video', 'Ταχύτητα', 'GPS', 'Ανάλυση οθόνης', 'Μπαταρία', 'Σχέση ποιότητας τιμής', 'Μουσική']

if aspect_list is None:
    print('Error: aspect_list is None')
else:
    for column in aspect_list:
        df[column] = '-' 
# fill the aspect list with 1, 0, -1 for pros, so-so, cons
import warnings
warnings.filterwarnings('ignore')
for i in tqdm(range(len(df)), desc="Processing rows"):
    temp = ast.literal_eval(df['agg_rating'].iloc[i])
    for j in temp['pros']:
        df[j].iloc[i] = 1
    for j in temp['so-so']:
        df[j].iloc[i] = 0
    for j in temp['cons']:
        df[j].iloc[i] = -1
df=df.drop(columns=['GPS'])
# uncomment ama den theloume na kratisoume tis pavles
#df = df[~df[aspect_list].isin(['-']).any(axis=1)]
'''
df["reviews"] = df["comment"].apply(spell_check)

# Apply lemmatization and preprocessing
df["text_lemma"] = df["reviews"].apply(lemmatize)
df["text_proc"] = df["text_lemma"].apply(lambda x: preprocess_text(x, stopwords))
'''
tqdm.pandas(desc="Spell checking comments")

df["reviews"] = df["comment"].progress_apply(spell_check)

tqdm.pandas(desc="Lemmatizing text")

df["text_lemma"] = df["reviews"].progress_apply(lemmatize)

tqdm.pandas(desc="Preprocessing text")

df["text_proc"] = df["text_lemma"].progress_apply(lambda x: preprocess_text(x, stopwords))
df.dropna(subset=["text_proc"], inplace=True)
df = df[df["text_proc"].astype(bool)]
#df.to_csv('test100_reviews.csv', index=False)  
dftest=df['text_proc']
summinmax = [0]
suffix_prune_el = 3  # prune in words
string_min_score = 0.76  # matching score
checkedWords = 0  # number of words that were checked
totalWords = 0  # sum of words


scorerev = [0]  # score per review
mins = [-1]  # min score per review
maxs = [1]  # max score per review
i=0
for review in tqdm(dftest,desc='apply sentiment algo'):

    #print(review)
 # bgazw to /n pou ebale to opencsv

    flag = False  # kathe review arxikopoiw false. An ginei true meta h epomenh leksh pou brisketai den metrietai
    
    rvwords = review.split(" ")  # kathe leksh pou exei to review

    rvwords = list(rvwords) 
    
    for words in rvwords:
        sr = 0  # sr start every word
        totalWords = totalWords + 1  # count words
        words = clean_accent(str(words))  # clean accent of word
    
        # emoticon first before any stiksh split so not to lose
        if words in emot:
            checkedWords = checkedWords + 1  # word find counter
            sr = scorem[emot.index(words)]
            scorerev[i] = scorerev[i] + sr  # if found adds score to review score
        else:
    
            # punctuation if no emoticon found
    
            a = [""]  # starts a dummy array to see if there is a !
            if "!" in words:
                a = words.split(
                    "!"
                )  # word is spliting from !. After this algorithm
                # cant find |! and word remains the same without !
                # so I can add word's score with ! boost
    
            for p in range(0, len(words)):
                if (
                    words[p : p + 1] in stikshh
                ):  # replacing every weird char with '' so word can be clear
                    words = words.replace(words[p : p + 1], "")
                    words = words.replace(".", "")
    
            # threepeat letters checker and hunspell sugestion after removing them.
            # Tested and gives good suggestions. Check also that word is not a punctuation or number
            k = [""]
            for p in range(3, len(words)):
    
                if words[p - 1 : p] == words[p - 2 : p - 1] == words[
                    p - 3 : p - 2
                ] and (words[p - 1 : p] not in stiksh):
                    words = "".join(sorted(set(words), key=words.index))
                    # print(words)
    
                    # k = h.suggest(words)
                    # if k != ():
                    #     words = k[0]
                    # break
    
            # Negative word check. If found flag=True and next word emotion skipped
            if words in neg:
                checkedWords = checkedWords + 1
                flag = True
    
            # main list check and scoring
            # get words that start with the first letter of word that we check
            # saves A LOT of time
            for wrd in [m for m in word if m.lower().startswith(words[:1])]:
                match = words.find(
                    wrd[: max(3, len(wrd) - suffix_prune_el)]
                )  # match word with pruning
                scorera = SequenceMatcher(
                    None, words, wrd
                ).ratio()  # ratio of final matching
                if match == 0 and scorera > string_min_score:  # match and ratio>
                    checkedWords = checkedWords + 1  # word counter
                    if flag == True:
                        flag = False  # if flag=True do it false and stop
                    else:
                        sr = score[word.index(wrd)]  # found score of word
                        if a[0] != "":  # If ! found
                            if sr == -1:  # score of word from -1->2
                                sr = 2
                            else:
                                sr = sr + 1  # other score of word +1
    
                        scorerev[i] = scorerev[i] + sr  # sum score of review
            # if words in boost add in score
            if words in boost:
                checkedWords = checkedWords + 1  # word counter
                sr = scorebo[boost.index(words)]
                scorerev[i] = scorerev[i] + sr
        # check for max review score	until this word in every case se is the added score
        # from word
        if sr > maxs[i]:
            maxs[i] = sr
        # check for min review score	until this word
        if sr < mins[i]:
            mins[i] = sr
    
    # add min and max to produce the final score and label
    # -1 if neg, 0 if neutr, 1 if positive
    summinmax[i] = maxs[i] + mins[i]
    
    
    
    if summinmax[i] >= 0:
        summinmax[i] = 1    
    elif summinmax[i] < 0:
        summinmax[i] = 0
    #print(summinmax[i])
    '''
    # 	summinmax[i]=0
    else:
        summinmax[i] = 1
    '''
    i = i + 1
    t = [ summinmax, mins, maxs]
    #print(t)
    ratio = checkedWords / totalWords
    if(i!=len(df)):
        summinmax.append(0)
        scorerev.append(0)
        mins.append(-1)
        maxs.append(1)
dfstars=df['stars']
dfstars = dfstars.map({4: 1, 5: 1, 3: 1, 1: 0, 2: 0})
df['sentiment_predicted'] = summinmax
df['sentiment_star'] = list(dfstars)
df.drop(columns=['title', 'date', 'verified_purchase','agg_rating'], inplace=True)
df.to_csv(f'test_{len(df)}_reviews.csv', index=False)  
