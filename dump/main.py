import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
nltk.data.path.append('/home/sujay1844/.local/share/nltk_data/')

essays = []

def lcs(l1, l2):
	# Preprocessing the strings
	s1 = word_tokenize(l1)
	s2 = word_tokenize(l2)
	# Lowercase
	s1 = [w.lower() for w in s1]
	s2 = [w.lower() for w in s2]
	# Removing punctuation
	table = str.maketrans('', '', string.punctuation)
	s1 = [w.translate(table) for w in s1]
	s2 = [w.translate(table) for w in s2]
	# Removing stopwords
	stop_words = set(stopwords.words('english'))
	s1 = [w for w in s1 if not w in stop_words]
	s2 = [w for w in s2 if not w in stop_words]

	# storing the dp values
	dp = [[None]*(len(s1)+1) for i in range(len(s2)+1)]

	for i in range(len(s2)+1):
		for j in range(len(s1)+1):
			if i == 0 or j == 0:
				dp[i][j] = 0
			elif s2[i-1] == s1[j-1]:
				dp[i][j] = dp[i-1][j-1]+1
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	return dp[len(s2)][len(s1)]

def get_score(orig, test):
	sent_o = sent_tokenize(orig)
	sent_p = sent_tokenize(test)

	# maximum length of LCS for a sentence in suspicious text
	max_lcs = 0
	sum_lcs = 0

	for i in sent_p:
		for j in sent_o:
			l = lcs(i, j)
			max_lcs = max(max_lcs, l)
			sum_lcs += max_lcs
			max_lcs = 0

	score = sum_lcs / len(word_tokenize(orig))
	return score

# Flask api
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/check')
def api():
	for essay in essays:
		for essay2 in essays:
			if essay != essay2:
				scores.append(get_score(essay, essay2), essay, essay2)
	return jsonify({
		'scores': sorted(iterable=scores, key=lambda x: x[0], reverse=True)[:3]
	})

@app.route('/add', methods=['POST'])
def add_essay():
	data = request.get_json()
	essays.append(data['essay'])
	return jsonify({'success': True})


if __name__ == '__main__':
	app.run(debug=True)