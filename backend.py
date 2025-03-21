# import shelve
# import re
# from pathlib import Path
# import os
# import nltk
# import numpy
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from pathlib import Path
# import re
# import numpy as np

# def read_file(filepath):
#     content = Path(filepath).read_text()
#     return str(content)

# def preprocess_and_get_features(filepath, stop_words, text=None):
#     if text is None:
#         text = read_file(filepath)
    
#     ps = PorterStemmer()
    
#     tokens = []
#     for word in word_tokenize(text):
#         if re.match(r'/\d+', word):
#             tokens.append(word)
#         else:
#             if re.match(r'\w+', word):
#                 tokens.append(word)
    
#     features = [
#         (
#             pos,
#             ''.join(ps.stem(word.lower()).split("-"))
#         )
#         for pos, word in enumerate(tokens)
#         if word.lower() not in stop_words
#     ]
    
#     return features

# def preprocess_query(query,stop_words_directory):
#     stop_word_file = read_file(stop_words_directory)
#     stop_words = word_tokenize(stop_word_file)
#     return preprocess_and_get_features('',stop_words,query)

# def preprocess_proximity_query(query):
#     items = query.split(" ")
#     new_items = []
#     new_items.append(items[0])
#     for i in range(1,len(items)):
#         if items[i-1].startswith("/"):
#             new_items.append(items[i])
#         else:
#             if not items[i].startswith("/"):
#                 new_items.append("/1")
#             new_items.append(items[i])   
    
#     stop_word_file = read_file("Stopword-List.txt")
#     stop_words = word_tokenize(stop_word_file)
#     l = preprocess_and_get_features("",stop_words," ".join(new_items))
#     stack = []
#     words = []
#     k_values = []
#     stack.append(l[0])
#     for i in range(1,len(l)):
#         top = stack[-1][1]
#         if l[i][1].startswith("/") and top.startswith("/"):
#             stack[-1] = (stack[-1][0],"/" + str(int(l[i][1][1:]) + int(top[1:])))
#         else:
#             stack.append(l[i])


#     for item in stack:
#         if item[1].startswith("/"):
#             k_values.append(int(item[1][1:]))
#         else:
#             words.append(item)

#     return words, k_values

# class QueryProcessor:
#     def __init__(self,preprocessor_function,proximity_processor):
#         self.packages_updated = False
#         self.preprocessor_function = preprocessor_function
#         self.proximity_processor = proximity_processor
#         pass

#     def intersect_search(self, postings_list):
#         if len(postings_list) == 0:
#             return np.array([])  # Return an empty result if there's nothing to process

#         if len(postings_list) == 1:
#             return np.array(list(postings_list[0][1].keys()))

#         p1 = list(postings_list[0][1].keys())
#         for p2 in postings_list[1:]:
#             if len(p2) < 2:  # Ensure there's at least one valid entry
#                 return np.array([])  # If any posting list is empty, intersection is empty

#             p2 = list(p2[1].keys())
#             answer_set = []
#             i, j = 0, 0
#             while i < len(p1) and j < len(p2):
#                 if p1[i] == p2[j]:
#                     answer_set.append(p1[i])
#                     i += 1
#                     j += 1
#                 elif p1[i] < p2[j]:
#                     i += 1
#                 else:
#                     j += 1
#             p1 = answer_set

#         return np.array(sorted(p1))

#     def positional_intersect(self, postings_list):
#         result = set()  
#         for i in range(len(postings_list) - 1):
#             _, doc1 = postings_list[i]
#             _, doc2 = postings_list[i + 1]

#             for doc_id1, positions1 in doc1.items():
#                 if doc_id1 in doc2:
#                     positions2 = doc2[doc_id1]
#                     positions1 = doc1[doc_id1]
#                     if isinstance(positions1[0],tuple):
#                         positions1 = positions1[1:]
                        
#                     if isinstance(positions2[0],tuple):
#                         positions2 = positions2[1:]
                    
#                     p1_index, p2_index = 0, 0
#                     while p1_index < len(positions1) and p2_index < len(positions2):
                        
#                         if positions2[p2_index] == positions1[p1_index] + 1:
#                             result.add(doc_id1)
#                             p1_index += 1
#                             p2_index += 1
#                         elif positions1[p1_index] < positions2[p2_index]:
#                             p1_index += 1
#                         else:
#                             p2_index += 1
#         return list(result)

#     def get_postings(self, p_query):
#         postings_list = []
#         try:
#             with shelve.open('Indexes/positional_index') as db:
#                 for pos, word in p_query:
#                     # Using .get() to avoid KeyError and handling missing words
#                     result = db.get(word, np.array([]))  # Empty result if word is not found
#                     postings_list.append(result)
#         except Exception as e:
#             print(f"Error accessing the positional index: {e}")
#         return postings_list


#     def positional_query(self,text):
#         processed_query = self.preprocessor_function(text,'Stopword-List.txt')
#         return numpy.array(self.positional_intersect(self.get_postings(processed_query)))

#     def proximity_positional_intersect(self,postings_list,k_list):
#         result = set()  
#         k_index = 0
#         for i in range(len(postings_list) - 1):
#             _, doc1 = postings_list[i]
#             _, doc2 = postings_list[i + 1]

#             for doc_id1, positions1 in doc1.items():
#                 if doc_id1 in doc2:
#                     positions2 = doc2[doc_id1]
#                     positions1 = doc1[doc_id1]
#                     if isinstance(positions1[0],tuple):
#                         positions1 = positions1[1:]
                        
#                     if isinstance(positions2[0],tuple):
#                         positions2 = positions2[1:]
                    
#                     p1_index, p2_index = 0, 0
#                     while p1_index < len(positions1) and p2_index < len(positions2):
                        
#                         if abs(positions2[p2_index] - positions1[p1_index]) <= k_list[k_index]:
#                             result.add(doc_id1)
#                             p1_index += 1
#                             p2_index += 1
#                         elif positions1[p1_index] < positions2[p2_index]:
#                             p1_index += 1
#                         else:
#                             p2_index += 1
#             k_index += 1
        
#         return list(result)

#     def proximity_query(self,text):
#         words, k_list = self.proximity_processor(text)
#         postings_list = self.get_postings(words)
#         return numpy.array(self.proximity_positional_intersect(postings_list,k_list))
              
#     def boolean_query_start(self, text):
#         p_query = self.preprocessor_function(text, 'Stopword-List.txt')
#         postings_list = self.get_postings(p_query)

#         if not postings_list or all(isinstance(pl, np.ndarray) and pl.size == 0 for pl in postings_list):
#             return np.array([])  # Ensure it doesn't fail on empty lists

#         postings_list_sorted = sorted(postings_list, key=lambda x: x[0] if len(x) > 0 else float('inf'))
#         return np.array(self.intersect_search(postings_list_sorted))

    
#     def boolean_query(self,bool_query):
#         subqueries = [x.strip() for x in bool_query.split("OR")]
#         result = numpy.array([])
#         for subQuery in subqueries:
#             q_ith = " ".join([x.strip() for x in subQuery.split("AND")])
#             result = numpy.union1d(result,self.boolean_query_start(q_ith))
#         return result

    
#     def evaluate_query(self, result_set, relevant_set,log=True):
#         precision = len(np.intersect1d(result_set, relevant_set)) / len(result_set) if len(result_set) > 0 else 0
#         recall = len(np.intersect1d(result_set, relevant_set)) / len(relevant_set) if len(relevant_set) > 0 else 0
#         if log:
#             print("Precision: ", precision)
#             print("Recall: ", recall)    
#             print("Actual Relevant: ")
#             print(relevant_set)
#             print("Relevant Retrieved: ")
#             print(np.intersect1d(result_set, relevant_set))
#             print("Irrelevant Retrieved: ")
#             print(np.setdiff1d(result_set, relevant_set))
#         return precision, recall
    
#     def execute_query(self,text,type):
#         if type == "boolean":
#             return self.boolean_query(text)
#         elif type == "phrase":
#             return self.positional_query(text)
#         elif type == "proximity":
#             return self.proximity_query(text)
#         else:
#             print("Not valid query type x :(")
#             return None
    

# # The code has two dependent files.
# # a folder named 'Indexes' containing necessary files
# # a file named 'Stopword-List.txt'
import shelve
import re
from pathlib import Path
import os
import nltk
import numpy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import re
import numpy as np

def read_file(filepath):
    try:
        content = Path(filepath).read_text()
        return str(content)
    except Exception as e:
        return f"Error in read_file: {str(e)}"

def preprocess_and_get_features(filepath, stop_words, text=None):
    try:
        if text is None:
            text = read_file(filepath)
            if "Error in" in text:
                return text  # Return the error message if read_file failed
        
        ps = PorterStemmer()
        
        tokens = []
        for word in word_tokenize(text):
            if re.match(r'/\d+', word):
                tokens.append(word)
            else:
                if re.match(r'\w+', word):
                    tokens.append(word)
        
        features = [
            (
                pos,
                ''.join(ps.stem(word.lower()).split("-"))
            )
            for pos, word in enumerate(tokens)
            if word.lower() not in stop_words
        ]
        
        return features
    except Exception as e:
        return f"Error in preprocess_and_get_features: {str(e)}"

def preprocess_query(query, stop_words_directory):
    try:
        stop_word_file = read_file(stop_words_directory)
        if "Error in" in stop_word_file:
            return stop_word_file  # Return the error message if read_file failed
        stop_words = word_tokenize(stop_word_file)
        return preprocess_and_get_features('', stop_words, query)
    except Exception as e:
        return f"Error in preprocess_query: {str(e)}"

def preprocess_proximity_query(query):
    try:
        items = query.split(" ")
        new_items = []
        new_items.append(items[0])
        for i in range(1, len(items)):
            if items[i - 1].startswith("/"):
                new_items.append(items[i])
            else:
                if not items[i].startswith("/"):
                    new_items.append("/1")
                new_items.append(items[i])   

        stop_word_file = read_file("Stopword-List.txt")
        if "Error in" in stop_word_file:
            return stop_word_file  # Return the error message if read_file failed
        stop_words = word_tokenize(stop_word_file)
        l = preprocess_and_get_features("", stop_words, " ".join(new_items))
        if "Error in" in l:
            return l  # Return the error message if preprocess_and_get_features failed

        stack = []
        words = []
        k_values = []
        stack.append(l[0])
        for i in range(1, len(l)):
            top = stack[-1][1]
            if l[i][1].startswith("/") and top.startswith("/"):
                stack[-1] = (stack[-1][0], "/" + str(int(l[i][1][1:]) + int(top[1:])))
            else:
                stack.append(l[i])

        for item in stack:
            if item[1].startswith("/"):
                k_values.append(int(item[1][1:]))
            else:
                words.append(item)

        return words, k_values
    except Exception as e:
        return f"Error in preprocess_proximity_query: {str(e)}"

class QueryProcessor:
    def __init__(self, preprocessor_function, proximity_processor):
        self.packages_updated = False
        self.preprocessor_function = preprocessor_function
        self.proximity_processor = proximity_processor

    def intersect_search(self, postings_list):
        try:
            if len(postings_list) == 0:
                return np.array([])  # Return an empty result if there's nothing to process

            if len(postings_list) == 1:
                return np.array(list(postings_list[0][1].keys()))

            p1 = list(postings_list[0][1].keys())
            for p2 in postings_list[1:]:
                if len(p2) < 2:  # Ensure there's at least one valid entry
                    return np.array([])  # If any posting list is empty, intersection is empty

                p2 = list(p2[1].keys())
                answer_set = []
                i, j = 0, 0
                while i < len(p1) and j < len(p2):
                    if p1[i] == p2[j]:
                        answer_set.append(p1[i])
                        i += 1
                        j += 1
                    elif p1[i] < p2[j]:
                        i += 1
                    else:
                        j += 1
                p1 = answer_set

            return np.array(sorted(p1))
        except Exception as e:
            return f"Error in intersect_search: {str(e)}"

    def positional_intersect(self, postings_list):
        try:
            result = set()  
            for i in range(len(postings_list) - 1):
                _, doc1 = postings_list[i]
                _, doc2 = postings_list[i + 1]

                for doc_id1, positions1 in doc1.items():
                    if doc_id1 in doc2:
                        positions2 = doc2[doc_id1]
                        positions1 = doc1[doc_id1]
                        if isinstance(positions1[0], tuple):
                            positions1 = positions1[1:]
                        
                        if isinstance(positions2[0], tuple):
                            positions2 = positions2[1:]
                        
                        p1_index, p2_index = 0, 0
                        while p1_index < len(positions1) and p2_index < len(positions2):
                            if positions2[p2_index] == positions1[p1_index] + 1:
                                result.add(doc_id1)
                                p1_index += 1
                                p2_index += 1
                            elif positions1[p1_index] < positions2[p2_index]:
                                p1_index += 1
                            else:
                                p2_index += 1
            return list(result)
        except Exception as e:
            return f"Error in positional_intersect: {str(e)}"

    def get_postings(self, p_query):
        try:
            postings_list = []
            with shelve.open('Indexes/positional_index') as db:
                for pos, word in p_query:
                    result = db.get(word, np.array([]))  # Empty result if word is not found
                    postings_list.append(result)
            return postings_list
        except Exception as e:
            return f"Error in get_postings: {str(e)}"

    def positional_query(self, text):
        try:
            processed_query = self.preprocessor_function(text, 'Stopword-List.txt')
            if "Error in" in processed_query:
                return processed_query  # Return the error message if preprocessing failed
            return numpy.array(self.positional_intersect(self.get_postings(processed_query)))
        except Exception as e:
            return f"Error in positional_query: {str(e)}"

    def proximity_positional_intersect(self, postings_list, k_list):
        try:
            result = set()  
            k_index = 0
            for i in range(len(postings_list) - 1):
                _, doc1 = postings_list[i]
                _, doc2 = postings_list[i + 1]

                for doc_id1, positions1 in doc1.items():
                    if doc_id1 in doc2:
                        positions2 = doc2[doc_id1]
                        positions1 = doc1[doc_id1]
                        if isinstance(positions1[0], tuple):
                            positions1 = positions1[1:]
                        
                        if isinstance(positions2[0], tuple):
                            positions2 = positions2[1:]
                        
                        p1_index, p2_index = 0, 0
                        while p1_index < len(positions1) and p2_index < len(positions2):
                            if abs(positions2[p2_index] - positions1[p1_index]) <= k_list[k_index]:
                                result.add(doc_id1)
                                p1_index += 1
                                p2_index += 1
                            elif positions1[p1_index] < positions2[p2_index]:
                                p1_index += 1
                            else:
                                p2_index += 1
                k_index += 1

            return list(result)
        except Exception as e:
            return f"Error in proximity_positional_intersect: {str(e)}"

    def proximity_query(self, text):
        try:
            words, k_list = self.proximity_processor(text)
            if "Error in" in words:  # Handle errors from proximity_processor
                return words  # Return the error message if preprocessing failed
            postings_list = self.get_postings(words)
            return numpy.array(self.proximity_positional_intersect(postings_list, k_list))
        except Exception as e:
            return f"Error in proximity_query: {str(e)}"

    def boolean_query_start(self, text):
        try:
            p_query = self.preprocessor_function(text, 'Stopword-List.txt')
            if "Error in" in p_query:  # Handle errors from preprocessing
                return p_query  # Return the error message if preprocessing failed
            postings_list = self.get_postings(p_query)

            if not postings_list or all(isinstance(pl, np.ndarray) and pl.size == 0 for pl in postings_list):
                return np.array([])  # Ensure it doesn't fail on empty lists

            postings_list_sorted = sorted(postings_list, key=lambda x: x[0] if len(x) > 0 else float('inf'))
            return np.array(self.intersect_search(postings_list_sorted))
        except Exception as e:
            return f"Error in boolean_query_start: {str(e)}"

    def boolean_query(self, bool_query):
        try:
            subqueries = [x.strip() for x in bool_query.split("OR")]
            result = numpy.array([])
            for subQuery in subqueries:
                q_ith = " ".join([x.strip() for x in subQuery.split("AND")])
                result = numpy.union1d(result, self.boolean_query_start(q_ith))
            return result
        except Exception as e:
            return f"Error in boolean_query: {str(e)}"

    def evaluate_query(self, result_set, relevant_set, log=True):
        try:
            precision = len(np.intersect1d(result_set, relevant_set)) / len(result_set) if len(result_set) > 0 else 0
            recall = len(np.intersect1d(result_set, relevant_set)) / len(relevant_set) if len(relevant_set) > 0 else 0
            if log:
                print("Precision: ", precision)
                print("Recall: ", recall)    
                print("Actual Relevant: ")
                print(relevant_set)
                print("Relevant Retrieved: ")
                print(np.intersect1d(result_set, relevant_set))
                print("Irrelevant Retrieved: ")
                print(np.setdiff1d(result_set, relevant_set))
            return precision, recall
        except Exception as e:
            return f"Error in evaluate_query: {str(e)}"

    def execute_query(self, text, query_type):
        try:
            if query_type == "boolean":
                return self.boolean_query(text)
            elif query_type == "phrase":
                return self.positional_query(text)
            elif query_type == "proximity":
                return self.proximity_query(text)
            else:
                print("Not valid query type x :(")
                return None
        except Exception as e:
            return f"Error in execute_query: {str(e)}"
