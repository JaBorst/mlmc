import numpy as np
import itertools
from sklearn.manifold import TSNE


def cooc_matrix(labels, classes):
    """Deprecated"""
    coocs = np.zeros((len(classes),len(classes)))
    frequencies = np.zeros((len(classes),1))
    for labelset in labels:
        for p in list(itertools.combinations(labelset, 2)):
            coocs[classes[p[0]],classes[p[1]]] += 1
            coocs[classes[p[1]],classes[p[0]]] += 1
            frequencies[classes[p[1]],0] +=1
            frequencies[classes[p[0]], 0] += 1

    return coocs, frequencies


def correlate_similarity(coocs, embeddings, n, classwise=False, corr="spearman"):
    """Deprecated"""
    cooc_rank = np.argsort(coocs, -1)[:, -n::-1]
    embed_rank = np.argsort(np.dot(embeddings, embeddings.transpose()), -1, )[:, -n::-1]

    from scipy.stats import spearmanr, kendalltau
    classcorrelations = []
    if corr=="spearman": fct = spearmanr
    if corr=="kendalltau": fct = kendalltau
    for  a, b in zip(cooc_rank, embed_rank):
        classcorrelations.append(fct(a, b)[0])

    if classwise:
        return classcorrelations
    else:
        return [np.mean(np.abs(classcorrelations)), np.std(np.abs(classcorrelations))],\
               [np.mean(classcorrelations), np.std(classcorrelations)]

def show_graph(vectors, classes):
    """
    Plot A TSNE
    :param vectors: Vectors to plot
    :param classes: Labels of the vectors to plot
    :return:
    """
    import matplotlib.pyplot as plt
    X_embedded = TSNE(n_components=2).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1])
    for i, txt in enumerate(classes.keys()):
        ax.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]))
    plt.show()

keywordmap = {"Sports": ["sport", "sports", "sporting"], "Business":["business"], "World": ["world","politics", "war"], "Sci/Tech": ["science", "technology"] ,
                    "Company":["company"], "EducationalInstitution": ["Education", "institution"], "Artist":["artist"],
                    "Athlete":["athlete"], "OfficeHolder":["officeholder"], "MeanOfTransportation": ["Transportation", "vehicle"],
                    "Building":["building"], "NaturalPlace":["nature", "region", "location"], "Village":["village"],
                    "Animal":["animal"], "Plant":["plant"], "Album":["album"], "Film":["film"], "WrittenWork":["writing", "literature", "work"],
                    "ABBR": ["abbreviation"], "DESC": ["description"], "ENTY":["entity", "person"], "HUM":["human", "person"], "LOC":["location"], "NUM": ["number"],
                    "Society & Culture":["society", "culture"], "Science & Mathematics":["science", "mathematics"], "Health":["health"],
                    "Education & Reference":["Education", "reference"], "Computers & Internet":["computer", "internet"], "Business & Finance": ["business", "finance"],
                    "Entertainment & Music":["entertainment", "music"], "Family & Relationships": ["family", "relationship"], "Politics & Government":["politics", "government"],
                    # "1":["1", "worst", "terrible"], "2":["2","poor", "odd", "simple"], "3":["3", "neutral","ok", "fine"], "4":["4", "bold", "worth", "good", "nice"], "5":["5","amazing", "excellent", "wow"],
                    "1":["1"], "2":["2"], "3":["3"], "4":["4",], "5":["5"],"one":["1"], "two":["2"], "three":["3"], "four":["4",], "five":["5"],
                    "negative":["1", "2"], "positive":["4","5"],
                    "ENTY:sport": ["entity", "sport"], "ENTY:dismed": ["entity","disease", "medicine"], "LOC:city": ["location", "city"],
                    "DESC:reason": ["description","reason"],
                    "NUM:other": ["number"],"LOC:state": ["location", "state"],"NUM:speed": ["number", "speed"],"NUM:ord": ["number", "order", "rank"],
                    "ENTY:event": ["entity","event"],"ENTY:substance": ["entity","element", "substance"],"NUM:perc": ["number", "percentage", "fraction"],
                    "ENTY:product": ["entity","product"],"ENTY:animal": ["entity","animal"],"DESC:manner": ["description", "manner", "action"],
                    "ENTY:cremat": ["entity","creative","invention","book"],"ENTY:color": ["entity","color"],"ENTY:techmeth": ["entity","technique", "method"],
                    "NUM:dist": ["number",  "distance", "measure"],"NUM:weight": ["number", "weight"],"LOC:mount": ["location", "mountain"],
                    "HUM:title": ["person", "title"],"HUM:gr": ["group", "organization", "person"],
                    "HUM:desc": ["person", "description"],"ABBR:abb": ["abbreviation"],
                    "ENTY:currency": ["entity","currency"],"DESC:def": ["description", "definition"],"NUM:code": ["number", "code"],"LOC:other": ["location"],
                    "ENTY:other": ["entity", "other"],"ENTY:body": ["entity","body", "organ"],"ENTY:instru": ["entity","music", "instrument"],
                    "ENTY:termeq": ["entity","synonym"],"NUM:money": ["number", "money", "price"],"NUM:temp": ["number", "temperature"],
                    "LOC:country": ["location", "country"],"ABBR:exp": ["abbreviation", "expression"],"ENTY:symbol": ["entity","symbol", "sign"],
                    "ENTY:religion":["entity" ,"religion"],"HUM:ind": ["individual", "person"],"ENTY:letter": ["entity","letter", "character"],
                    "NUM:date": ["number", "date"],"ENTY:lang": ["entity","language"],"ENTY:veh": ["entity","vehicle"],
                    "NUM:count": ["number", "count"],"ENTY:word": ["entity","word", "special", "property"],"NUM:period": ["number", "time period", "time"],
                    "ENTY:plant": ["entity","plant"],"ENTY:food": ["entity", "food"],"NUM:volsize": ["number", "volume", "size"],
                    "DESC:desc": ["description"],
                    'Teen & Young Adult Mystery & Suspense': ['teen', 'young', 'adult', 'mystery', 'suspense'],
                    'Regency Romance': ['regency', 'romance'], 'Baking & Desserts': ['baking', 'dessert'],
                    'Teen & Young Adult Fantasy Fiction': ['teen', 'young', 'adult', 'fantasy', 'fiction'],
                    'Travel: Australia & Oceania': ['travel', 'Australia', 'Oceania'],
                    'Children’s Chapter Books': ['child', 'chapter', 'book'],
                    'Science Fiction': ['science', 'fiction'], 'Travel: Europe': ['travel', 'Europe'],
                    'North American World History': ['North American', 'world', 'history'],
                    'Graphic Novels & Manga': ['graphic', 'novel', 'anime'],
                    'Travel: Central & South America': ['travel', 'central', 'South America'],
                    'Performing Arts': ['performing arts'], '20th Century U.S. History': ['20th', 'century', 'U.S.', 'history'],
                    'Arts & Entertainment Biographies & Memoirs': ['arts', 'entertainment', 'biography', 'memoir'],
                    'Travel: Caribbean & Mexico': ['travel', 'Caribbean', 'Mexico'], 'Photography': ['photography'],
                    'Alternative Therapies': ['alternative', 'therapy'], 'Nonfiction Classics': ['nonfiction', 'classics'],
                    'Paranormal Fiction': ['paranormal', 'fiction'], 'Military Fiction': ['military', 'fiction'],
                    'Colonial/Revolutionary Period': ['colonial', 'revolutionary', 'period'], 'Politics': ['politics'],
                    'Test Preparation': ['test', 'preparation'],
                    'Teen & Young Adult Historical Fiction': ['teen', 'young', 'adult', 'historical', 'fiction'],
                    'U.S. History': ['U.S.', 'history'], 'Children’s Picture Books': ['child', 'picture', 'book'],
                    'Fiction Classics': ['fiction', 'classics'], 'Ancient World History': ['ancient', 'world', 'history'],
                    'Classics': ['classics'], 'Military Science Fiction': ['military', 'science fiction'],
                    'World War I Military History': ['world war', 'i', 'military', 'history'],
                    'Fiction': ['fiction'], 'Paranormal Romance': ['paranormal', 'romance'], 'Women’s Fiction': ['woman', 'fiction'],
                    'Crime Mysteries': ['crime', 'mystery'], 'Design': ['design'], 'Personal Growth': ['personal', 'growth'],
                    'Marketing': ['marketing'], 'Travel': ['travel'], 'Personal Finance': ['personal', 'finance'], 'Crafts & Hobbies': ['craft', 'hobby'],
                    'Religion & Philosophy': ['religion', 'philosophy'], 'Cooking Methods': ['cooking', 'method'],
                    'Fairy Tales': ['fairy tale'], 'Travel Writing': ['travel', 'writing'],
                    'Nonfiction': ['nonfiction'], 'Science': ['science'], 'World History': ['world', 'history'],
                    'Fantasy': ['fantasy'], 'Language': ['language'], 'Technology': ['technology'],
                    'Western Romance': ['western', 'romance'], 'Biography & Memoir': ['biography', 'memoir'],
                    'Home & Garden': ['home', 'garden'], 'Contemporary Fantasy': ['contemporary', 'fantasy'],
                    'Spiritual Fiction': ['spiritual', 'fiction'], 'Suspense Romance': ['suspense', 'romance'],
                    'Art': ['art'], 'Literary Figure Biographies & Memoirs': ['literary', 'figure', 'biography', 'memoir'],
                    'Mystery & Suspense': ['mystery', 'suspense'], 'Civil War Period': ['civil war', 'period'],
                    'Teen & Young Adult Fiction': ['teen', 'young', 'adult', 'fiction'],
                    'Historical Figure Biographies & Memoirs': ['historical', 'figure', 'biography', 'memoir'],
                    'African World History': ['African', 'world', 'history'], 'Health & Reference': ['health', 'reference'],
                    'Childrens Media Tie-In Books': ['child', 'audio', 'video', 'tie-in', 'book'],
                    'Regional & Ethnic Cooking': ['regional', 'ethnic', 'cooking'], 'Romance': ['romance'],
                    'Children’s Boxed Sets': ['child', 'boxed', 'set'], 'Teen & Young Adult Social Issues': ['teen', 'young', 'adult', 'social', 'issue'],
                    'Economics': ['economics'], 'Wine & Beverage': ['wine', 'beverage'],
                    'Latin American World History': ['Latin American', 'world', 'history'], 'Pets': ['pet'], 'Music': ['music'],
                    'Urban Fantasy': ['urban', 'fantasy'], 'Travel: Africa': ['travel', 'Africa'],
                    'Health & Fitness': ['health', 'fitness'], 'Noir Mysteries': ['film noir', 'mystery'],
                    'Children’s Middle Grade Historical Books': ['child', 'middle', 'grade', 'historical', 'book'],
                    'Children’s Middle Grade Fantasy & Magical Books': ['child', 'middle', 'grade', 'fantasy', 'magical', 'book'],
                    'Children’s Books': ['child', 'book'], 'Teen & Young Adult Action & Adventure': ['teen', 'young', 'adult', 'action', 'adventure'],
                    'Historical Romance': ['historical', 'romance'], 'Teen & Young Adult': ['teen', 'young', 'adult'],
                    'Children’s Middle Grade Mystery & Detective Books': ['child', 'middle', 'grade', 'mystery', 'detective', 'book'],
                    'Epic Fantasy': ['epic', 'fantasy'], 'Humor': ['humor'], 'Self-Improvement': ['self-improvement'],
                    'Historical Fiction': ['historical', 'fiction'], 'Games': ['game'], 'Literary Criticism': ['literary', 'criticism', 'literary criticism'],
                    'Asian World History': ['Asian', 'world', 'history'], 'Middle Eastern World History': ['Middle Eastern', 'world', 'history'],
                    'Teen & Young Adult Romance': ['teen', 'young', 'adult', 'romance'], 'Writing': ['writing'],
                    'Travel: USA & Canada': ['travel', 'USA', 'Canada'], 'Literary Fiction': ['literary', 'fiction'], 'Popular Science': ['popular', 'science'],
                    'Literary Collections': ['literary', 'collection'], 'Children’s Middle Grade Books': ['child', 'middle', 'grade', 'book'],
                    'Erotica': ['erotica'], 'Crafts, Home & Garden': ['craft', 'home', 'garden'],
                    'Children’s Middle Grade Action & Adventure Books': ['child', 'middle', 'grade', 'action', 'adventure', 'book'],
                    'Travel: Asia': ['travel', 'Asia'], 'Cooking': ['cooking'], 'Cozy Mysteries': ['cozy', 'mystery'], 'History': ['history'],
                    'Espionage Mysteries': ['espionage', 'mystery'], 'World War II Military History': ['world war', 'ii', 'military', 'history'],
                    'Suspense & Thriller': ['suspense', 'thriller'], '19th Century U.S. History': ['19th', 'century', 'U.S.', 'history'],
                    'Space Opera': ['space', 'opera'], 'Diet & Nutrition': ['diet', 'nutrition'], 'Religion': ['religion'],
                    'Arts & Entertainment': ['art', 'entertainment'], 'Specialty Travel': ['specialty', 'travel'],
                    'Weddings': ['wedding'], 'Teen & Young Adult Nonfiction': ['teen', 'young', 'adult', 'nonfiction'],
                    '21st Century U.S. History': ['21st', 'century', 'U.S.', 'history'], 'Gothic & Horror': ['gothic', 'horror'],
                    'Domestic Politics': ['domestic', 'politics'], 'Reference': ['reference'], 'Beauty': ['beauty'],
                    'Western Fiction': ['Western', 'fiction'],
                    'Teen & Young Adult Science Fiction': ['teen', 'young', 'adult', 'science fiction'],
                    'Philosophy': ['philosophy'], 'Parenting': ['parent', 'raise'],
                    'Native American History': ['Native American', 'history'], 'Poetry': ['poetry'], 'Psychology': ['psychology'],
                    'Inspiration & Motivation': ['inspiration', 'motivation'], 'Step Into Reading': ['beginner', "start", 'reading'],
                    'Exercise': ['exercise'], 'Bibles': ['bible'], 'Travel: Middle East': ['travel', 'Middle East'],
                    'New Adult Romance': ['new', 'adult', 'romance'], 'Contemporary Romance': ['contemporary', 'romance'],
                    'Military History': ['military', 'history'], 'Cyber Punk': ["cyberpunk"],
                    'Children’s Middle Grade Sports Books': ['child', 'middle', 'grade', 'sport', 'book'],
                    'European World History': ['European', 'world', 'history'],
                    'Political Figure Biographies & Memoirs': ['political', 'figure', 'biography', 'memoir'],
                    'Children’s Activity & Novelty Books': ['child', 'activity', 'novelty', 'book'],
                    '1950 – Present Military History': ['present', 'military', 'history'],
                    'Children’s Board Books': ['child', 'board', 'book'], 'World Politics': ['world', 'politics'],
                    'Food Memoir & Travel': ['food', 'memoir', 'travel'], 'Management': ['management'],
                    'alt.atheism':["atheism"], 'comp.graphics': ["computer", "graphics", "computer graphics"],
                    'comp.os.ms-windows.misc': ['computer',"operating system", "Windows"],
                    'comp.sys.ibm.pc.hardware': ['computer', 'system', 'IBM' , 'hardware'],
                    'comp.sys.mac.hardware':  ['computer', 'system', 'Apple', "Mac" , 'hardware'], 'comp.windows.x': ["computer", "Windows"],
                    'misc.forsale': ["for sale", "sale"],
                    'rec.autos': ["recreational", "auto", "car"], 'rec.motorcycles': ["recreational", "motorcycle"],
                    'rec.sport.baseball': ["recreational", "sport", "baseball"],
                    'rec.sport.hockey': ["recreational", "sport", "hockey"],
                    'sci.crypt': ["science", "cryptography"],
                    'sci.electronics': ["science", "electronics"],
                    'sci.med': ["science", "medicine"],
                    'sci.space': ["science", "outer space"],
                    'soc.religion.christian': ["society", "religion", "christianity"],
                    'talk.politics.guns': ["politic", "gun"],
                    'talk.politics.mideast': ["politic", "Middle East"],
                    'talk.politics.misc': ["politic"],
                    'talk.religion.misc':["religion"],
              }