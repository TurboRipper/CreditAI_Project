"""CreditAgent local research application. Run: python app.py"""
import csv
import io
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from credit_engine import CreditEngine, FEATURES, FIELD_SPECS, validate

BASE = Path(__file__).resolve().parent
DATASET = Path(os.environ.get('CREDIT_DATASET', str(BASE/'german.data')))
STATE = Path(os.environ.get('CREDIT_STATE_DIR', str(BASE/'runtime')))
STATE.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2*1024*1024
engine = None
startup_error = None
engine_lock = threading.Lock()
tracking = {'status':'Starting', 'message':'Waiting for model training.'}
mlclient = None
experiment_id = None

@contextmanager
def database():
    db = sqlite3.connect(STATE/'credit_agent.db', timeout=30)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

with database() as db:
    db.executescript('''CREATE TABLE IF NOT EXISTS assessments (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, input_json TEXT NOT NULL,
    result_json TEXT NOT NULL, reviewer_decision TEXT NOT NULL DEFAULT 'Pending',
    reviewer_reason TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL DEFAULT '');
    CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id TEXT NOT NULL, created_at TEXT NOT NULL, reviewer TEXT NOT NULL,
    decision TEXT NOT NULL, reason TEXT NOT NULL, risk_agreement TEXT NOT NULL);''')


def track(name, params, metrics):
    global tracking
    if mlclient is None:
        return
    run_id = None
    try:
        run = mlclient.create_run(experiment_id, tags={'mlflow.runName':name})
        run_id = run.info.run_id
        for key,value in params.items(): mlclient.log_param(run_id,key,value)
        for key,value in metrics.items(): mlclient.log_metric(run_id,key,float(value))
        mlclient.set_terminated(run_id)
        tracking = {'status':'Active','message':'Local MLflow experiment logging is active.', 'last_run_id':run_id}
    except Exception as exc:
        tracking = {'status':'Unavailable','message':str(exc)}
        if run_id:
            try: mlclient.set_terminated(run_id, status='FAILED')
            except Exception: pass


def initialize():
    global engine, startup_error, mlclient, experiment_id, tracking
    try:
        trained = CreditEngine(DATASET)
        try:
            from mlflow import MlflowClient
            uri = 'sqlite:///' + (STATE/'mlflow.db').as_posix()
            mlclient = MlflowClient(tracking_uri=uri)
            experiment = mlclient.get_experiment_by_name('CreditAgent')
            experiment_id = experiment.experiment_id if experiment else mlclient.create_experiment('CreditAgent', artifact_location=(STATE/'artifacts').as_uri())
            track('Training '+trained.model_version, {'model':trained.best_model_name,'version':trained.model_version,
                'dataset_sha256':trained.dataset_hash,'threshold':trained.threshold,'selection':'3-fold training ROC-AUC',
                'threshold_selection':'validation FP+5FN'}, {k:trained.summary[k] for k in ['accuracy','precision','recall','f1']})
            for row in trained.summary['comparison']:
                track('CV '+row['model'], {'model':row['model'],'version':trained.model_version}, {k:v for k,v in row.items() if k!='model'})
        except Exception as exc:
            tracking = {'status':'Unavailable','message':str(exc)}
        engine = trained
    except Exception as exc:
        startup_error = str(exc)
        print('STARTUP ERROR:', startup_error, flush=True)


@app.before_request
def local_mutation_guard():
    if request.method == 'POST':
        # Local browser UI sends this header. Cross-origin pages cannot silently post forms.
        if request.headers.get('X-CreditAgent') != 'local-ui':
            return jsonify(error='Missing local application header.'),403
        origin = request.headers.get('Origin')
        if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
            return jsonify(error='Cross-origin requests are not accepted.'),403
    if request.path.startswith('/api/') and request.path not in ['/api/health','/api/schema'] and engine is None:
        return jsonify(error=startup_error or 'Models are still training. Please wait.'),503


@app.errorhandler(Exception)
def errors(exc):
    if isinstance(exc, HTTPException): return jsonify(error=exc.description),exc.code
    if isinstance(exc, (ValueError, KeyError, TypeError, UnicodeError)):
        return jsonify(error=str(exc)),400
    app.logger.exception('Request failed')
    return jsonify(error='The request failed. Check the terminal for details.'),500


def body():
    value = request.get_json()
    if not isinstance(value,dict): raise ValueError('Request must be a JSON object.')
    return value


def get_record(identifier):
    with database() as db:
        row = db.execute('SELECT * FROM assessments WHERE id=?',(identifier,)).fetchone()
        events = db.execute('SELECT * FROM reviews WHERE assessment_id=? ORDER BY id',(identifier,)).fetchall()
    if row is None:
        from werkzeug.exceptions import NotFound
        raise NotFound('Assessment not found.')
    return {**dict(row), 'input':json.loads(row['input_json']), 'result':json.loads(row['result_json']), 'reviews':[dict(e) for e in events]}


def assess_and_save(customer):
    customer = validate(customer)
    with engine_lock: result = engine.assess(customer)
    result['assessment_id'] = 'CA-'+uuid.uuid4().hex[:12].upper()
    with database() as db:
        db.execute('INSERT INTO assessments (id,created_at,input_json,result_json) VALUES (?,?,?,?)',
            (result['assessment_id'],result['created_at'],json.dumps(customer),json.dumps(result)))
    track(result['assessment_id'], {'version':result['model_version'],'threshold':result['threshold']},
        {k:result[k] for k in ['bad_probability','confidence','latency_ms']})
    return result


@app.get('/')
def index(): return render_template('index.html')

@app.get('/api/health')
def health(): return jsonify(status='error' if startup_error else ('ready' if engine else 'training'),error=startup_error)

@app.get('/api/schema')
def schema(): return jsonify(FIELD_SPECS)

@app.get('/api/summary')
def summary(): return jsonify(engine.summary)

@app.post('/api/assess')
def assess(): return jsonify(assess_and_save(body()))

@app.post('/api/what-if')
def what_if():
    payload=body()
    record=get_record(payload.get('assessment_id',''))
    if record['result']['model_version'] != engine.model_version:
        raise ValueError('This assessment uses an older model. Run a new assessment first.')
    return jsonify(engine.what_if(record['input'],payload.get('changes')))

@app.get('/api/fairness')
def fairness(): return jsonify(engine.fairness)

@app.get('/api/history')
def history():
    with database() as db:
        rows=db.execute('SELECT * FROM assessments ORDER BY created_at DESC LIMIT 100').fetchall()
    return jsonify([{k:v for k,v in dict(row).items() if k not in ['input_json','result_json']} | {'result':json.loads(row['result_json'])} for row in rows])

@app.get('/api/assessment/<identifier>')
def assessment(identifier):
    record=get_record(identifier)
    record.pop('input_json'); record.pop('result_json')
    return jsonify(record)

@app.post('/api/review/<identifier>')
def review(identifier):
    get_record(identifier)
    payload=body()
    decision=payload.get('decision')
    if decision not in ['Continue review','Request documents','Approve after human review','Decline after human review']:
        raise ValueError('Choose a valid reviewer decision.')
    reason,reviewer=payload.get('reason'),payload.get('reviewer')
    if not isinstance(reason,str) or not 3<=len(reason.strip())<=2000:
        raise ValueError('Enter a reviewer reason between 3 and 2000 characters.')
    if not isinstance(reviewer,str) or not 2<=len(reviewer.strip())<=80:
        raise ValueError('Enter a reviewer name between 2 and 80 characters.')
    agreement=payload.get('risk_agreement')
    if agreement not in ['Agree','Disagree','Uncertain']: raise ValueError('Choose agreement with the AI risk label.')
    with database() as db:
        db.execute('UPDATE assessments SET reviewer_decision=?,reviewer_reason=?,reviewer=? WHERE id=?',
            (decision,reason.strip(),reviewer.strip(),identifier))
        db.execute('INSERT INTO reviews (assessment_id,created_at,reviewer,decision,reason,risk_agreement) VALUES (?,?,?,?,?,?)',
            (identifier,datetime.now(timezone.utc).isoformat(),reviewer.strip(),decision,reason.strip(),agreement))
    return jsonify(status='saved')

@app.get('/api/monitoring')
def monitoring():
    with database() as db:
        rows=db.execute('SELECT input_json,result_json,reviewer_decision FROM assessments ORDER BY created_at DESC').fetchall()
        reviews=db.execute('SELECT * FROM reviews WHERE id IN (SELECT MAX(id) FROM reviews GROUP BY assessment_id)').fetchall()
    results=[json.loads(r['result_json']) for r in rows]
    determinate=[r for r in reviews if r['risk_agreement']!='Uncertain']
    current=[r for r in rows if json.loads(r['result_json'])['model_version']==engine.model_version]
    average=lambda key: round(sum(r[key] for r in results)/len(results),2) if results else None
    return jsonify(assessment_count=len(rows),pending=sum(r['reviewer_decision']=='Pending' for r in rows),
        bad_risk_rate=round(sum(r['prediction'] for r in results)/len(results)*100,2) if results else None,
        average_confidence=average('confidence'),average_latency_ms=average('latency_ms'),
        disagreement_rate=round(sum(r['risk_agreement']=='Disagree' for r in determinate)/len(determinate)*100,2) if determinate else None,
        reviewed=len(reviews),agreement_denominator=len(determinate),tracking=tracking,model_version=engine.model_version,
        drift=engine.drift([json.loads(r['input_json']) for r in current[:200]]))

@app.get('/api/batch-template')
def batch_template():
    output=io.StringIO(); writer=csv.DictWriter(output,fieldnames=FEATURES); writer.writeheader()
    writer.writerow({f['name']:f.get('value',f.get('options',[[None]])[0][0]) for f in FIELD_SPECS})
    return send_file(io.BytesIO(output.getvalue().encode()),mimetype='text/csv',as_attachment=True,download_name='credit_batch_template.csv')

@app.post('/api/batch')
def batch():
    upload=request.files.get('file')
    if not upload: raise ValueError('Choose a CSV file.')
    reader=csv.DictReader(io.StringIO(upload.read().decode('utf-8-sig')))
    if reader.fieldnames is None or len(reader.fieldnames)!=17 or set(reader.fieldnames)!=set(FEATURES):
        raise ValueError('CSV must have exactly the 17 unique columns in the template.')
    rows=list(reader)
    if not 1<=len(rows)<=100: raise ValueError('Upload between 1 and 100 rows.')
    results=[]
    for number,row in enumerate(rows,1):
        try:
            result=assess_and_save(row)
            results.append({'row':number,'assessment_id':result['assessment_id'],'risk':result['risk_label'],'bad_probability':result['bad_probability']})
        except ValueError as exc: results.append({'row':number,'error':str(exc)})
    return jsonify(results)

@app.get('/api/report/<identifier>')
def report(identifier):
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font_dir = Path(reportlab.__file__).parent/'fonts'
    if 'CreditVera' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('CreditVera', str(font_dir/'Vera.ttf')))
        pdfmetrics.registerFont(TTFont('CreditVeraBold', str(font_dir/'VeraBd.ttf')))
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.lib.pagesizes import A4
    record=get_record(identifier); result=record['result']
    stream=io.BytesIO(); styles=getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = 'CreditVeraBold' if style.name in ['Title','Heading2'] else 'CreditVera'
    styles.add(ParagraphStyle(name='SmallBody',fontName='CreditVera',fontSize=9,leading=13,spaceAfter=6))
    p=lambda text: Paragraph(escape(str(text)),styles['SmallBody'])
    story=[Paragraph('CreditAgent / Assessment report',styles['Title']),Spacer(1,16)]
    for key,value in [('Assessment',identifier),('Created',result['created_at']),('Model',result['model']),('Version',result['model_version']),
        ('Risk label',result['risk_label']),('Bad-risk probability',f"{result['bad_probability']:.2f}%"),
        ('Good-risk probability',f"{result['good_probability']:.2f}%"),('Predicted-class probability',f"{result['confidence']:.2f}%"),
        ('Decision threshold',result['threshold']),('Recommendation',result['recommendation'])]: story.append(p(f'{key}: {value}'))
    story.extend([Spacer(1,12),Paragraph('Customer inputs',styles['Heading2'])])
    for spec in FIELD_SPECS:
        value=record['input'][spec['name']]
        if spec['type']=='select': value=dict(spec['options']).get(value,value)
        story.append(p(f"{spec['label']}: {value}"))
    explanation=result['explanation']
    story.extend([Spacer(1,12),Paragraph('SHAP explanation',styles['Heading2'])])
    if explanation['status']=='ready':
        story.append(p('Units: '+explanation['units']+'. Positive contributions increase modeled bad risk; these are associations, not causal effects.'))
        for item in explanation['contributions']: story.append(p(f"{item['feature']}: {item['impact']:+.5f}"))
    else: story.append(p('SHAP unavailable: '+explanation.get('message','')))
    story.extend([Spacer(1,12),Paragraph('Human review audit trail',styles['Heading2'])])
    if not record['reviews']: story.append(p('Pending human review.'))
    for event in record['reviews']:
        story.extend([p(f"{event['created_at']} / {event['reviewer']} / {event['decision']}"),p('Reason: '+event['reason']),p('Agreement with AI risk: '+event['risk_agreement'])])
    story.extend([Spacer(1,12),Paragraph('Workflow and limitations',styles['Heading2']),p(' > '.join(result['agent_trace'])),
        p(result['cost_status']),p('Historical German Credit research prototype. Currency is DM, not INR. Predicted-class probability is not calibrated certainty. This report does not establish loan eligibility or fairness. Final decisions require human review.')])
    def footer(canvas,doc):
        canvas.setFont('CreditVera',8);canvas.setFillColor(colors.HexColor('#686375'))
        canvas.drawString(42,25,'CreditAgent | Research decision support');canvas.drawRightString(A4[0]-42,25,f'Page {doc.page}')
    SimpleDocTemplate(stream,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=45,title=identifier).build(story,onFirstPage=footer,onLaterPages=footer)
    stream.seek(0)
    return send_file(stream,mimetype='application/pdf',as_attachment=True,download_name=identifier+'.pdf')

if __name__=='__main__':
    threading.Thread(target=initialize,daemon=True).start()
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','5000')),debug=False,threaded=True)
