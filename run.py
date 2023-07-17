import os
from sys import exit
from flask import Flask
from flask import url_for, render_template, flash, request, redirect, send_from_directory, abort, send_file
from werkzeug.utils import secure_filename
import glob
#
from modules import parse_config, populate_config, ConfigForm
# temp until analysis can be in modules
from analysis import NetCDFAnalysis

# powershell: $env:FLASK_APP = "run"
# bash: export FLASK_APP=run
# flask run

app = Flask(__name__)

'''
APP CONFIGURATION
'''
UPLOAD_FOLDER = './uploads'
UPLOAD_C_FOLDER = './uploads/config'
UPLOAD_A_FOLDER = './uploads/analysis'
ALLOWED_EXTENSIONS = {'txt', 'nc', 'cfg', 'json'}
app.config['UPLOAD_C_FOLDER'] = UPLOAD_C_FOLDER
app.config['UPLOAD_A_FOLDER'] = UPLOAD_A_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)
app.add_url_rule("/uploads/config/<name>", endpoint="download_c_file", build_only=True)
app.add_url_rule("/uploads/analysis/<name>", endpoint="download_a_file", build_only=True)

SECRET_KEY = "please_dont_hack_us_thanks"
app.config['SECRET_KEY'] = SECRET_KEY

'''
Internal Functions
'''
# this function courtesy of https://flask.palletsprojects.com/
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def c_dir_listing():
    filenames = os.listdir(UPLOAD_C_FOLDER)
    clean_files = []
    for i in filenames:
        if i[-4:] == ".cfg":
            clean_files.append(i)
        elif i[-5:] == ".json":
            clean_files.append(i)
    return clean_files

def a_dir_listing():
    filenames = os.listdir(UPLOAD_A_FOLDER)
    #filenames = glob.glob(os.path.join(UPLOAD_A_FOLDER,"*.nc"))
    clean_files = []
    for i in filenames:
        if i[-3:] == ".nc":
            clean_files.append(i)
    return clean_files

'''
ROUTES
'''
@app.route('/')
def landingpage():
    return redirect(url_for('homepage'))
    
@app.route('/home')
def homepage():
    return render_template("home.html")

@app.route('/config', methods=['POST', 'GET'])
def config():
    if c_dir_listing():
        filenames = c_dir_listing()
    else:
        filenames = None
    return render_template("config.html", filenames= filenames)

@app.route('/analysis', methods=['POST', 'GET'])
def analysis():
    if a_dir_listing():
        filenames = a_dir_listing()
    else:
        filenames = None
    return render_template("analysis.html", filenames= filenames)

# this function structure courtesy of https://flask.palletsprojects.com/
@app.route('/config/upload', methods=['GET', 'POST'])
def upload_conf_file():
    if c_dir_listing():
        filenames = c_dir_listing()
    else:
        filenames = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_C_FOLDER'], filename))
            return redirect(url_for('upload_conf_file', name=filename, filenames = filenames))
    return render_template('upload_conf.html', filenames = filenames)

@app.route('/analysis/upload', methods=['GET', 'POST'])
def upload_analysis_file():
    if a_dir_listing():
        filenames = a_dir_listing()
    else:
        filenames = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_A_FOLDER'], filename))
            return redirect(url_for('upload_analysis_file', name=filename, filenames=filenames))
    return render_template('upload_analysis.html', filenames=filenames)

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

@app.route('/uploads/config/<name>')
def download_c_file(name):
    return send_from_directory(app.config["UPLOAD_C_FOLDER"], name)

@app.route('/uploads/analysis/<name>')
def download_a_file(name):
    return send_from_directory(app.config["UPLOAD_A_FOLDER"], name)

@app.route('/config/configjson', methods=['GET', 'POST'])
def engage_jsonconf():
    found = False
    submitted = False
    if c_dir_listing():
        filenames = c_dir_listing()
        if "config.json" in filenames:
            found = True
    if found:
        form = ConfigForm()
        if form.validate_on_submit():
            populate_config(form.proc_batch.data, form.stream_batch.data, form.flow_method.data, form.flow_winsize.data, form.sparsematch_quality.data, form.pointsred_maxdist.data, form.pointsred_winsize.data)
            submitted = True
        form.proc_batch.data, form.stream_batch.data, form.flow_method.data, form.flow_winsize.data, form.sparsematch_quality.data, form.pointsred_maxdist.data, form.pointsred_winsize.data = parse_config(UPLOAD_C_FOLDER+"/config.json")
    return render_template('configjson.html', found=found, form=form, submitted=submitted)

@app.route('/analysis/proc', methods=['GET', 'POST'])
def analyse_netcdf():
    fname = False
    if a_dir_listing():
        filenames = a_dir_listing()
        for i in filenames:
            if i[-3:] == ".nc":
                fname = i
        if request.method == 'POST':
            return redirect(url_for('analysis_report'))
    return render_template('analyzenc.html', fname=fname)

@app.route('/analysis/report', methods=['GET', 'POST'])
def analysis_report():
    fname = ''
    if a_dir_listing():
        filenames = a_dir_listing()
        for i in filenames:
            if i[-3:] == ".nc":
                fname = i
    nca = NetCDFAnalysis(UPLOAD_A_FOLDER+"/"+fname)
    title, framedata, timeseriedata, spectrumdata, dirspectrumdata, hs, hm0, pp, Tm01, location, date, time, ncfile, duration, fps, meta, outfile = nca.analyse()
    dirspectrumdata1, dirspectrumdata2, dirspectrumdata3, dirspectrumdata4, dirspectrumdata5, dirspectrumdata6, dirspectrumdata7, dirspectrumdata8, dirspectrumdata9, dirspectrumdata10 = dirspectrumdata
    meta = meta.replace("- ", "\t\t- ")
    meta = meta.split("\n")
    ncfile = ncfile.split("(")[0]
    return render_template('analysis_temp.html', title=title, framedata=framedata, timeseriedata=timeseriedata, spectrumdata=spectrumdata, hs=hs, hm0=hm0, pp=pp, Tm01=Tm01, location=location, date=date, time=time, ncfile=ncfile, duration=duration, fps=fps, meta=meta, outfile=outfile, dirspectrumdata1=dirspectrumdata1, dirspectrumdata2=dirspectrumdata2, dirspectrumdata3=dirspectrumdata3, dirspectrumdata4=dirspectrumdata4, dirspectrumdata5=dirspectrumdata5, dirspectrumdata6=dirspectrumdata6, dirspectrumdata7=dirspectrumdata7, dirspectrumdata8=dirspectrumdata8, dirspectrumdata9=dirspectrumdata9, dirspectrumdata10=dirspectrumdata10)

@app.route('/downloadanalysis')
def return_analysis_output():
	try:
		return send_file(os.path.join(app.config['UPLOAD_A_FOLDER'], "nc_analysis.html"), as_attachment=True, attachment_filename='nc_analysis.html')
	except Exception as e:
		return str(e)
'''
Error Handling
'''
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run()