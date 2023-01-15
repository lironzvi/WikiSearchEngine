from flask import Flask, request, jsonify
import re
from nltk.stem.porter import PorterStemmer
from inverted_index_gcp import InvertedIndex
import RankingAndEvaluation
from RankingAndEvaluation import BM25_from_index
import os
import json
import nltk
from nltk.corpus import stopwords
import time
import pickle


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
ps = PorterStemmer()
all_stopwords = english_stopwords.union(corpus_stopwords)
global pr
global pv
global body_index
global title_index
global anchor_index
body_index = InvertedIndex.read_index('', 'body_index')
title_index = InvertedIndex.read_index('', 'title_index')
anchor_index = InvertedIndex.read_index('', 'anchor_index')
body_index_stemming = InvertedIndex.read_index('', 'stemming_body_index_new')
title_index_stemming = InvertedIndex.read_index('', 'stemming_title_index')
bm25_for_body = BM25_from_index(body_index_stemming,'body_locs_stemming')

# os.system("gsutil cp gs://titles_for_json/titles.json")
with open('titles.json') as titlesJSON:
    titles = json.load(titlesJSON)

with open('ranking1.json') as rankingJSON:
    pr = json.load(rankingJSON)

with open('pageviews-202108-user.pkl', 'rb') as f:
    pv = pickle.loads(f.read())


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    body_weight = 0.4
    title_weight = 0.2
    anchor_weight = 0.01
    pr_weight = 0.1
    pv_weight = 0.3
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    tokens = []
    for token in RE_WORD.finditer(query.lower()):
        if token.group() not in all_stopwords:
            tokens.append(token.group())

    stemming = [ps.stem(t) for t in tokens if t not in all_stopwords]
    print(tokens)
    if len(tokens) > 1:
        sorted_scores_body_as_list = bm25_for_body.search({1: stemming}, 100)[1]
    else:
        sorted_scores_body_as_list = RankingAndEvaluation.get_topN_score_for_queries(tokens, body_index, 'body_locs', 100)[1]
    sorted_scores_body = {}
    documents_and_scores_title = {}
    documents_and_scores_anchor = {}
    for doc_id, score in sorted_scores_body_as_list:
        sorted_scores_body[doc_id] = score

    for token in stemming:
        token_posting = InvertedIndex.read_posting_list(title_index_stemming,token,'title_locs_stemming')
        for doc, tf in token_posting:
            if doc not in documents_and_scores_title:
                documents_and_scores_title[doc] = 1/len(tokens)
            else:
                documents_and_scores_title[doc] += 1/len(tokens)
    documents_and_scores_title_list = [(documentId,document_score) for documentId,document_score in documents_and_scores_title.items()]
    documents_and_scores_title = dict(sorted(documents_and_scores_title_list,key=lambda x: x[1], reverse=True)[:100])

    documents_and_scores_anchor_as_list = RankingAndEvaluation.get_topN_score_for_queries(tokens, anchor_index,'anchor_locs', 100)[1]

    for doc_id, score in documents_and_scores_anchor_as_list:
        documents_and_scores_anchor[doc_id] = score

    all_docs = set()
    all_docs.update(list(sorted_scores_body.keys()))
    all_docs.update(list(documents_and_scores_title.keys()))
    all_docs.update(list(documents_and_scores_anchor.keys()))

    body_top_score = 1
    title_top_score = 1
    anchor_top_score = 1

    if sorted_scores_body.values():
        body_top_score = max(sorted_scores_body.values())
    if documents_and_scores_title.values():
        title_top_score = max(documents_and_scores_title.values())
    if documents_and_scores_anchor.values():
        anchor_top_score = max(documents_and_scores_anchor.values())

    page_rank_list = []
    page_view_list = []
    for doc in all_docs:
        doc_string = str(doc)
        if doc_string in pr:
            page_rank_list.append(pr[doc_string])
        if doc_string in pv:
            page_view_list.append(pv[doc_string])

    pv_top_score = 1
    pr_top_score = 1
    if len(page_view_list) > 0:
        pv_top_score = max(page_view_list)
    if len(page_rank_list) > 0:
        pr_top_score = max(page_rank_list)

    docs_scores = {}

    for doc in all_docs:
        body_score = anchor_score = title_score = pr_score = pv_score = 0
        if doc in sorted_scores_body:
            body_score = body_weight * (sorted_scores_body[doc] / body_top_score)
        if doc in documents_and_scores_title:
            title_score = title_weight * (documents_and_scores_title[doc] / title_top_score)
        # if doc in documents_and_scores_anchor:
        #     anchor_score = anchor_weight * (documents_and_scores_anchor[doc] / anchor_top_score)
        if str(doc) in page_rank_list:
            pr_score = pr_weight * (page_rank_list[doc] / pr_top_score)
        if str(doc) in page_view_list:
            pv_score = pv_weight * (page_view_list[doc] / pv_top_score)

        docs_scores[doc] = body_score + title_score + anchor_score + pr_score + pv_score

    sorted_scores_list = sorted(list(docs_scores.items()), key=lambda x: x[1], reverse=True)[:100]
    sorted_scores_list = [(a[0], a[1]) for a in sorted_scores_list if str(a[0]) in titles]
    tmp_map = map(lambda a: (a[0], titles[str(a[0])]), sorted_scores_list)
    res = list(tmp_map)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # adding tokens to list
    tokens = []
    for token in RE_WORD.finditer(query.lower()):
        if token.group() not in all_stopwords:
            tokens.append(token.group())

    res = RankingAndEvaluation.get_topN_score_for_queries(tokens, body_index, 'body_locs', 100)
    tmp_map = map(lambda a: (a[0], titles[str(a[0])]), res[1])
    res = list(tmp_map)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = []
    for token in RE_WORD.finditer(query.lower()):
        if token.group() not in all_stopwords:
            tokens.append(token.group())
    docs = {}
    for token in set(tokens):
        iter = InvertedIndex.read_posting_list(title_index, token, 'title_locs')
        for doc, tf in iter:
            if docs.get(doc) is not None:
                docs[doc] = docs[doc] + (1 / len(set(tokens)))
            else:
                docs[doc] = (1 / len(set(tokens)))
    lst = []
    for key in docs:
        lst.append(tuple((key, docs[key])))
    sorted_scores_list = sorted(lst, key=lambda x: x[1], reverse=True)
    tmp_map = map(lambda a: (a[0], titles[str(a[0])]), sorted_scores_list)
    res = list(tmp_map)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
        # BEGIN SOLUTION
    tokens = []
    for token in RE_WORD.finditer(query.lower()):
        if token.group() not in all_stopwords:
            tokens.append(token.group())
    docs = {}
    for token in set(tokens):
        iter = InvertedIndex.read_posting_list(anchor_index, token, 'anchor_locs')
        for doc, tf in iter:
            if docs.get(doc) is not None:
                docs[doc] = docs[doc] + (1 / len(set(tokens)))
            else:
                docs[doc] = (1 / len(set(tokens)))
    lst = []
    for key in docs:
        lst.append(tuple((key, docs[key])))
    sorted_scores_list = sorted(lst, key=lambda x: x[1], reverse=True)
    tmp_map = map(lambda a: (a[0], titles[str(a[0])]), sorted_scores_list)
    res = list(tmp_map)

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pr[str(i)] for i in wiki_ids if str(i) in pr.keys]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [pv[str(i)] for i in wiki_ids if str(i) in pv.keys]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
