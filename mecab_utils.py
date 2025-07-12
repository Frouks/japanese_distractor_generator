import MeCab
from collections import Counter 

def verifyInstallation(): 
    """Checks if the MeCab installation was successfull or not"""
    # Create a simple wakati-style tagger (space-separated words)
    wakati = MeCab.Tagger("-Owakati")
    result = wakati.parse("pythonが好きです")
    #.split()
    print(result)
    # Should output: ['python', 'が', '好き', 'です']

    # Standard tagger with more details
    tagger = MeCab.Tagger()
    print(tagger.parse("pythonが好きです"))
    # Should show detailed morphological information

def extract_parts(text, pos_type):
    """Extract all words of a specific part of speech from text.
    
    Parameters:
        text (str): The Japanese text to analyze
        pos_type (str): The part of speech to extract. Default is "名詞" (noun).
                        Common values include:
                        - "名詞": Nouns (e.g., 大学, 研究)
                        - "動詞": Verbs (e.g., 食べる, 行く)
                        - "形容詞": Adjectives (e.g., 美しい)
                        - "助詞": Particles (e.g., は, が, を)
                        - "形状詞": Adjectival nouns (e.g., 好き)
    
    Returns:
        list: A list of strings containing all words of the specified part of speech
    """
    tagger = MeCab.Tagger()
    # Instead of getting the full analysis text, this gets a linked list of nodes where each node represents one token (word)
    node = tagger.parseToNode(text)
    results = []
    
    while node:
        if node.surface and node.feature.split(',')[0].startswith(pos_type):
            results.append(node.surface)
        node = node.next
        
    return results

def word_frequency(text):
    """Calculate the frequency of each word in the given text.
    
    Parameters:
        text (str): The Japanese text to analyze
    
    Returns:
        Counter: A Counter object mapping words to their frequency counts,
                where keys are words and values are their occurrence counts.
                Can be accessed like a dictionary or using Counter methods
                like most_common().
    """
    tagger = MeCab.Tagger("-Owakati")
    words = tagger.parse(text).strip().split()
    return Counter(words)
