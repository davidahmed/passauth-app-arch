from flask import Flask, redirect, url_for, request
from flask import render_template
import os

app = Flask(__name__)

entry = ""

@app.route('/index')
@app.route('/')
def hello():
	if entry:
		print('Next one')
	return render_template('index.html')

@app.route('/submit',methods = ['POST', 'GET'])
def post():
	if request.method == 'POST':
		content = request.get_json(silent=True)
		
		with open(content['username'] + '.txt', 'a+') as f:
			f.write(str(content))
		
		return redirect(url_for('hello'))
	else:
		return redirect(url_for('index'))


if __name__ == '__main__':

	app.run(debug=True)
