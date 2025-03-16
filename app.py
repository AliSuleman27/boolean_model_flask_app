from flask import Flask, render_template, request, jsonify, redirect, url_for
from code_1 import QueryProcessor, preprocess_query, preprocess_proximity_query, read_file
import os
import nltk

nltk.data.path.append('data/nltk_data')
qp = QueryProcessor(preprocessor_function=preprocess_query, proximity_processor=preprocess_proximity_query)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def my_app():
    context = {'total_results': 0, 'docs': []}  # Ensure context always exists

    if request.method == "POST":
        query = request.form['search_text']
        query_type = request.form['query_type']

        if "/" in query and  query_type != "proximity":
            context["error"] = "The query is not a properly formatted " + query_type + " query"
            
        else:
            if qp:
                results = qp.execute_query(query, query_type)
                if "Error in" in results:
                    context["error"] = results
                    context["query"] = query
                else:
                    docs = []

                    for result in results:
                        doc_path = str(int(result))
                        file_content = read_file("Abstracts/" + doc_path + ".txt").strip().split("\n", 1)
                        title = file_content[0] if len(file_content) > 0 else "Untitled"
                        text = file_content[1] if len(file_content) > 1 else ""

                        docs.append({'doc_path':doc_path,'title': title, 'text': text})

                    context = {  
                        'total_results': len(results),
                        'docs': docs,
                        'query_type': query_type,
                        'query' : query
                    }
        
    return render_template('index.html', context=context)


@app.route('/document/<doc_path>', methods=['GET'])
def getDoc(doc_path):
    doc_path = os.path.join('Abstracts', f"{doc_path}.txt")
    
    context = {'title': '', 'keywords': '', 'abstract': ''}
    
    if not os.path.exists(doc_path):
        context['error'] = 'Document Not Found'
    else:
        file_content = read_file(doc_path).strip().split("\n\n", 2)
        context['title'] = file_content[0] if len(file_content) > 0 else "Untitled"
        context['keywords'] = file_content[1] if len(file_content) > 1 else ""
        context['abstract'] = file_content[2] if len(file_content) > 2 else ""
    
    # return jsonify(context)
    return render_template('document.html', context=context)


if __name__ == '__main__':
    app.run(debug=True)
