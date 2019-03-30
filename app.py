from flask import Flask, redirect, url_for, request
from flask import render_template
import os, time

app = Flask(__name__)

common_unames_file = open('./data/common_unames.txt', 'r')
common_unames = common_unames_file.read().split()

attempts = {}

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submission')
@app.route('/submission/<int:attempts>',  methods = ['GET'])
def submission(attempts=None):
	return render_template('index.html', attempts=attempts)

@app.route('/submit', methods = ['POST', 'GET'])
def post():
	if request.method == 'POST':
		content = request.get_json(silent=True)

		if content['username'] in common_unames:
			return url_for('submission', attempts=1000)

		content['headers'] = request.headers
		content['time'] = int(time.time())
		with open('./data/' + content['username'] + '.txt', 'a+') as f:
			f.write(str(content))

		if content['username'] in attempts:
			attempts[content['username']] += 1
		else:
			attempts[content['username']] = 1

		return url_for('submission', attempts=attempts[content['username']])

	else:
		return redirect(url_for('index'))


if __name__ == '__main__':

	app.run(debug=True, host='0.0.0.0', port=80)
