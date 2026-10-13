import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing_extensions import TypedDict

COLUMNS = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount",
 "savings_account", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
 "residence_since", "property", "age", "other_installment_plans", "housing", "existing_credits",
 "job", "people_liable", "telephone", "foreign_worker", "target"]
FEATURES = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount",
 "savings_account", "employment_since", "installment_rate", "other_debtors", "residence_since",
 "property", "other_installment_plans", "housing", "existing_credits", "job", "people_liable", "telephone"]
CATEGORICAL = ["checking_account", "credit_history", "purpose", "savings_account", "employment_since",
 "other_debtors", "property", "other_installment_plans", "housing", "job", "telephone"]
NUMERICAL = ["duration_months", "credit_amount", "installment_rate", "residence_since", "existing_credits", "people_liable"]



FIELD_SPECS = [{'name': 'checking_account', 'label': 'Checking account', 'type': 'select', 'options': [['A11', 'Below 0 DM'], ['A12', '0 to below 200 DM'], ['A13', '200 DM or more'], ['A14', 'No checking account']]}, {'name': 'duration_months', 'label': 'Loan duration (months)', 'type': 'number', 'min': 4, 'max': 72, 'value': 24}, {'name': 'credit_history', 'label': 'Credit history', 'type': 'select', 'options': [['A30', 'No credit / paid duly'], ['A31', 'Bank credits paid duly'], ['A32', 'Existing credits paid duly'], ['A33', 'Past payment delay'], ['A34', 'Critical account / credits elsewhere']]}, {'name': 'purpose', 'label': 'Loan purpose', 'type': 'select', 'options': [['A40', 'New car'], ['A41', 'Used car'], ['A42', 'Furniture / equipment'], ['A43', 'Radio / television'], ['A44', 'Domestic appliances'], ['A45', 'Repairs'], ['A46', 'Education'], ['A47', 'Vacation'], ['A48', 'Retraining'], ['A49', 'Business'], ['A410', 'Other']]}, {'name': 'credit_amount', 'label': 'Credit amount (DM)', 'type': 'number', 'min': 250, 'max': 20000, 'value': 3000}, {'name': 'savings_account', 'label': 'Savings account', 'type': 'select', 'options': [['A61', 'Below 100 DM'], ['A62', '100–499 DM'], ['A63', '500–999 DM'], ['A64', '1000 DM or more'], ['A65', 'Unknown / none']]}, {'name': 'employment_since', 'label': 'Employment duration', 'type': 'select', 'options': [['A71', 'Unemployed'], ['A72', 'Less than 1 year'], ['A73', '1 to less than 4 years'], ['A74', '4 to less than 7 years'], ['A75', '7 years or more']]}, {'name': 'installment_rate', 'label': 'Installment rate (% of disposable income)', 'type': 'number', 'min': 1, 'max': 4, 'value': 2}, {'name': 'other_debtors', 'label': 'Other debtor / guarantor', 'type': 'select', 'options': [['A101', 'None'], ['A102', 'Co-applicant'], ['A103', 'Guarantor']]}, {'name': 'residence_since', 'label': 'Years at current residence (1–4)', 'type': 'number', 'min': 1, 'max': 4, 'value': 2}, {'name': 'property', 'label': 'Property', 'type': 'select', 'options': [['A121', 'Real estate'], ['A122', 'Life insurance / society'], ['A123', 'Car / other'], ['A124', 'Unknown / none']]}, {'name': 'other_installment_plans', 'label': 'Other installment plans', 'type': 'select', 'options': [['A141', 'Bank'], ['A142', 'Stores'], ['A143', 'None']]}, {'name': 'housing', 'label': 'Housing', 'type': 'select', 'options': [['A151', 'Rent'], ['A152', 'Own'], ['A153', 'Free']]}, {'name': 'existing_credits', 'label': 'Existing credits (1–4)', 'type': 'number', 'min': 1, 'max': 4, 'value': 1}, {'name': 'job', 'label': 'Job', 'type': 'select', 'options': [['A171', 'Unemployed / unskilled'], ['A172', 'Unskilled resident'], ['A173', 'Skilled employee'], ['A174', 'Management / self-employed']]}, {'name': 'people_liable', 'label': 'Financial dependents (1–2)', 'type': 'number', 'min': 1, 'max': 2, 'value': 1}, {'name': 'telephone', 'label': 'Telephone', 'type': 'select', 'options': [['A191', 'No'], ['A192', 'Yes']]}]

import hashlib
import json
import math
import time
import os
import sys
import tempfile
import joblib
from importlib.metadata import version as package_version
from threadpoolctl import threadpool_limits
from pathlib import Path
from datetime import datetime, timezone

VERSION = 'creditagent-2.1-fast'

# Fast replacement for CreditAgent Pro's engine; app.py and the UI stay unchanged.
# Fixed small models replace grid search. Models are cached after local training.
# Never put downloaded/untrusted .joblib files in this application's cache folder.
# Set CREDIT_RETRAIN=1 to bypass the cache for one launch, then unset it.

class CreditState(TypedDict, total=False):
    customer_data: dict
    validated_data: dict
    model_input: object
    prediction: int
    bad_probability: float
    good_probability: float
    confidence: float
    risk_label: str
    cost_status: str
    explanation: dict
    recommendation: str
    agent_trace: list


def validate(customer):
    if not isinstance(customer, dict):
        raise ValueError('Customer input must be a JSON object.')
    extra = set(customer) - set(FEATURES)
    if extra:
        raise ValueError('Unexpected fields: ' + ', '.join(sorted(str(key) for key in extra)))
    clean = {}
    for field in FIELD_SPECS:
        key, label = field['name'], field['label']
        value = customer.get(key)
        if value is None or isinstance(value, bool) or str(value).strip() == '':
            raise ValueError(f'Please enter {label}.')
        if field['type'] == 'number':
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{label} must be a number.') from None
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f'{label} must be a finite whole number.')
            if not field['min'] <= number <= field['max']:
                raise ValueError(f"{label} must be between {field['min']} and {field['max']}.")
            clean[key] = int(number)
        else:
            if value not in [x[0] for x in field['options']]:
                raise ValueError(f'Choose a valid {label}.')
            clean[key] = value
    return clean


class CreditEngine:
    def __init__(self, dataset_path):
        started = time.perf_counter()
        path = Path(dataset_path)
        raw = pd.read_csv(path, sep=r'\s+', header=None)
        if raw.shape[1] != 21 or raw.isna().any().any():
            raise ValueError('Use the original german.data: 21 space-separated columns, no header.')
        raw.columns = COLUMNS
        if not raw.target.isin([1, 2]).all():
            raise ValueError('Dataset target must contain 1 (good) and 2 (bad).')
        self.data = raw.drop_duplicates()
        X, y = self.data[FEATURES], self.data.target.map({1: 0, 2: 1})
        X_dev, self.X_test, y_dev, self.y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)
        self.dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.best_model_name, self.model, comparison = self._train()
        prob = self.model.predict_proba(self.X_val)[:, 1]
        candidates = []
        for threshold in np.linspace(.01, .99, 99):
            prediction = (prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_val, prediction, labels=[0, 1]).ravel()
            candidates.append((int(fp + 5*fn), -f1_score(self.y_val, prediction, zero_division=0), float(threshold)))
        cost, _, self.threshold = min(candidates)
        test_prob = self.model.predict_proba(self.X_test)[:, 1]
        test_pred = (test_prob >= self.threshold).astype(int)
        import sklearn
        self.model_version = VERSION + '-' + self.dataset_hash[:10] + '-sk' + sklearn.__version__
        self.summary = dict(model=self.best_model_name, model_version=self.model_version,
            rows=len(self.data), features=17, threshold=round(self.threshold, 2),
            accuracy=round(accuracy_score(self.y_test, test_pred)*100, 2),
            precision=round(precision_score(self.y_test, test_pred, zero_division=0)*100, 2),
            recall=round(recall_score(self.y_test, test_pred, zero_division=0)*100, 2),
            f1=round(f1_score(self.y_test, test_pred, zero_division=0)*100, 2),
            comparison=comparison, validation_cost=cost, cost_ratio='FN:FP = 5:1',
            train_rows=len(self.X_train), validation_rows=len(self.X_val), test_rows=len(self.X_test),
            cache_status=self.cache_status, training_mode='Fixed small models; 3-fold CV; no grid search')
        self.explainer = None
        self.explanation_error = None
        self._prepare_explainer()
        self.fairness = self._fairness_audit()
        self.graph = self._build_graph()
        self.summary['engine_startup_seconds'] = round(time.perf_counter()-started, 3)
        print(f"Engine ready in {self.summary['engine_startup_seconds']:.2f}s ({self.cache_status}).", flush=True)

    def _train(self):
        # Cache only this application's locally fitted model, not LangGraph/SHAP objects.
        # Dataset, source, Python and dependency changes invalidate the cache.
        dependencies = {p: package_version(p) for p in ['scikit-learn','numpy','scipy','pandas','joblib']}
        signature = dict(dataset=self.dataset_hash, source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         version=VERSION, python=sys.version, dependencies=dependencies)
        cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        cache_root = Path(os.environ.get('CREDIT_STATE_DIR', str(Path(__file__).resolve().parent/'runtime')))/'model_cache'
        cache_file = cache_root/(cache_key+'.joblib')
        self.cache_status = 'trained'
        if os.environ.get('CREDIT_RETRAIN') != '1' and cache_file.is_file():
            try:
                # joblib is pickle-based: use only the trusted cache created here.
                saved = joblib.load(cache_file)
                if saved.get('signature') == signature:
                    name, model, comparison = saved['name'], saved['model'], saved['comparison']
                    self.cache_status = 'loaded from cache'
                    print('Loaded saved model; skipping training and cross-validation.', flush=True)
                    return name, model, comparison
            except Exception as exc:
                print(f'Cache could not be read; training again: {exc}', flush=True)

        # Three fixed lightweight candidates, not a hyperparameter grid.
        # Training-only CV still selects the best model; test data stays untouched.
        candidates = {
            'Logistic Regression': LogisticRegression(C=1.0, solver='liblinear', max_iter=300, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_leaf=4, n_jobs=1, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=30, max_depth=1, learning_rate=.1, random_state=42)}
        models, comparison = {}, []
        for name, estimator in candidates.items():
            print('Training:', name, flush=True)
            prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
                ('num', StandardScaler() if name == 'Logistic Regression' else 'passthrough', NUMERICAL)])
            pipe = Pipeline([('prep', prep), ('model', estimator)])
            with threadpool_limits(limits=1):
                scores = cross_validate(pipe, self.X_train, self.y_train,
                    scoring={'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'auc': 'roc_auc'},
                    cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1, error_score='raise')
                pipe.fit(self.X_train, self.y_train)
            models[name] = pipe
            comparison.append({'model': name, **{m: round(float(np.mean(scores['test_'+m]))*100, 2) for m in ['accuracy', 'recall', 'f1', 'auc']}})
        best = max(comparison, key=lambda r: (r['auc'], r['f1']))['model']
        temporary = None
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='training-', suffix='.tmp', dir=cache_root)
            os.close(fd)
            joblib.dump(dict(signature=signature, name=best, model=models[best], comparison=comparison), temporary, compress=3)
            os.replace(temporary, cache_file)
            print('Saved model cache for faster restarts.', flush=True)
        except Exception as exc:
            self.cache_status = 'trained; cache save unavailable'
            print(f'Model is usable, but caching failed: {exc}', flush=True)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
        return best, models[best], comparison

    def _prepare_explainer(self):
        try:
            import shap
            prep, estimator = self.model.named_steps['prep'], self.model.named_steps['model']
            background = prep.transform(self.X_train.sample(min(80, len(self.X_train)), random_state=42))
            if hasattr(estimator, 'coef_'):
                self.explainer = shap.LinearExplainer(estimator, background)
                self.shap_units = 'log-odds'
            else:
                self.explainer = shap.TreeExplainer(estimator, data=background, feature_perturbation='interventional', model_output='probability')
                self.shap_units = 'probability points'
            self.feature_names = list(prep.get_feature_names_out())
        except Exception as exc:
            self.explanation_error = f'{type(exc).__name__}: {exc}'
            print('SHAP unavailable:', self.explanation_error, flush=True)

    def explain(self, frame):
        if self.explainer is None:
            return {'status': 'unavailable', 'message': self.explanation_error, 'contributions': []}
        try:
            transformed = self.model.named_steps['prep'].transform(frame)
            explanation = self.explainer(transformed)
            values = np.asarray(explanation.values)
            base = np.asarray(explanation.base_values)
            if values.ndim == 3:
                values = values[0, :, 1]
                baseline = float(base[0, 1])
            else:
                values = values[0]
                baseline = float(base.reshape(-1)[0])
            # Aggregate one-hot columns back to the original 17 input fields.
            totals = dict.fromkeys(FEATURES, 0.0)
            for name, value in zip(self.feature_names, values):
                clean = name.split('__', 1)[1]
                source = next(f for f in sorted(FEATURES, key=len, reverse=True) if clean == f or clean.startswith(f+'_'))
                totals[source] += float(value)
            scale = 100 if self.shap_units == 'probability points' else 1
            rows = [{'feature': k, 'impact': round(v*scale, 5)} for k, v in totals.items()]
            rows.sort(key=lambda r: abs(r['impact']), reverse=True)
            return {'status': 'ready', 'method': 'SHAP', 'units': self.shap_units,
                'baseline': baseline*scale, 'explained_output': (baseline + float(values.sum()))*scale,
                'background_rows': min(80, len(self.X_train)), 'contributions': rows}
        except Exception as exc:
            return {'status': 'unavailable', 'message': f'{type(exc).__name__}: {exc}', 'contributions': []}

    @staticmethod
    def _trace(state, name):
        return state.get('agent_trace', []) + [name]

    def _build_graph(self):
        graph = StateGraph(CreditState)
        nodes = [('Validation', self._validation), ('Preprocessing', self._preprocessing), ('Risk prediction', self._risk),
                 ('Cost analysis', self._cost), ('SHAP explanation', self._explanation), ('Recommendation', self._decision)]
        previous = START
        for name, function in nodes:
            graph.add_node(name, function)
            graph.add_edge(previous, name)
            previous = name
        graph.add_edge(previous, END)
        return graph.compile()

    def _validation(self, state):
        return {'validated_data': validate(state['customer_data']), 'agent_trace': self._trace(state, 'Validation')}

    def _preprocessing(self, state):
        return {'model_input': pd.DataFrame([state['validated_data']], columns=FEATURES), 'agent_trace': self._trace(state, 'Preprocessing')}

    def _risk(self, state):
        good, bad = self.model.predict_proba(state['model_input'])[0]
        pred = int(bad >= self.threshold)
        return {'prediction': pred, 'good_probability': float(good*100), 'bad_probability': float(bad*100),
            'confidence': float((bad if pred else good)*100), 'risk_label': 'Higher risk' if pred else 'Lower risk',
            'agent_trace': self._trace(state, 'Risk prediction')}

    def _cost(self, state):
        return {'cost_status': f'Validation-selected threshold {self.threshold:.2f}; missed bad-risk cases cost 5 units, false alerts cost 1 unit.',
            'agent_trace': self._trace(state, 'Cost analysis')}

    def _explanation(self, state):
        explanation = self.explain(state['model_input'])
        status = 'SHAP explanation' if explanation['status'] == 'ready' else 'SHAP unavailable'
        return {'explanation': explanation, 'agent_trace': self._trace(state, status)}

    def _decision(self, state):
        text = 'Request a detailed human credit review.' if state['prediction'] else 'Continue to the standard human credit review.'
        return {'recommendation': text, 'agent_trace': self._trace(state, 'Recommendation')}

    def assess(self, customer):
        start = time.perf_counter()
        state = self.graph.invoke({'customer_data': customer, 'agent_trace': []})
        result = {k: state[k] for k in ['prediction','good_probability','bad_probability','confidence','risk_label','cost_status','explanation','recommendation','agent_trace']}
        return {**result, 'model': self.best_model_name, 'model_version': self.model_version,
            'threshold': self.threshold, 'created_at': datetime.now(timezone.utc).isoformat(),
            'latency_ms': round((time.perf_counter()-start)*1000, 2)}

    def what_if(self, original, changes):
        allowed = {'credit_amount', 'duration_months', 'savings_account', 'installment_rate'}
        if not isinstance(changes, dict) or not changes or set(changes)-allowed:
            raise ValueError('Change amount, duration, savings or installment rate only.')
        before, after = validate(original), validate({**original, **changes})
        probs = self.model.predict_proba(pd.DataFrame([before, after], columns=FEATURES))[:, 1]*100
        return {'original_bad_probability': float(probs[0]), 'new_bad_probability': float(probs[1]),
            'difference': float(probs[1]-probs[0]), 'new_risk_label': 'Higher risk' if probs[1]/100 >= self.threshold else 'Lower risk',
            'note': 'Hypothetical model comparison; not a causal effect or a saved application.'}

    def _fairness_audit(self):
        rows = self.data.loc[self.X_test.index].copy()
        rows['truth'] = self.y_test
        rows['prediction'] = (self.model.predict_proba(self.X_test)[:, 1] >= self.threshold).astype(int)
        # Sex inferred from historical combined personal-status codes; not self-reported gender.
        rows['age_group'] = pd.cut(rows.age, [0,29,49,float('inf')], labels=['Under 30','30–49','50+'])
        rows['recorded_sex'] = rows.personal_status_sex.map({'A91':'Male','A92':'Female','A93':'Male','A94':'Male','A95':'Female'})
        rows['foreign_worker_group'] = rows.foreign_worker.map({'A201':'Yes','A202':'No'})
        groups, gaps = [], []
        for attribute in ['age_group','recorded_sex','foreign_worker_group']:
            attribute_rows = []
            for group, subset in rows.groupby(attribute, observed=True):
                tn, fp, fn, tp = confusion_matrix(subset.truth, subset.prediction, labels=[0,1]).ravel()
                ratio = lambda a,b: round(float(a/b*100),2) if b else None
                item = dict(attribute=attribute, group=str(group), samples=len(subset), positives=int(tp+fn), negatives=int(tn+fp),
                    accuracy=ratio(tp+tn,len(subset)), recall=ratio(tp,tp+fn), fpr=ratio(fp,fp+tn), fnr=ratio(fn,fn+tp),
                    small_sample=bool(len(subset)<30 or min(tp+fn,tn+fp)<10))
                groups.append(item); attribute_rows.append(item)
            gap = {'attribute': attribute}
            for metric in ['accuracy','recall','fpr','fnr']:
                values = [r[metric] for r in attribute_rows if r[metric] is not None]
                gap[metric] = round(max(values)-min(values),2) if len(values)>1 else None
            gaps.append(gap)
        return {'groups': groups, 'gaps': gaps, 'note': 'Offline held-out audit. Differences are descriptive, not proof of fairness. Small groups are flagged; undefined rates appear as N/A. Sensitive fields are excluded from prediction, but proxies may remain.'}

    def drift(self, applications):
        if len(applications) < 30:
            return {'status':'Collecting data', 'samples':len(applications), 'minimum':30, 'features':[]}
        recent = pd.DataFrame(applications)
        features = []
        for field in FEATURES:
            if field in NUMERICAL:
                cuts = np.unique(np.quantile(self.X_train[field], [.2,.4,.6,.8]))
                bins = np.r_[-np.inf, cuts, np.inf]
                expected = np.histogram(self.X_train[field], bins=bins)[0]
                actual = np.histogram(recent[field], bins=bins)[0]
            else:
                categories = sorted(set(self.X_train[field]) | set(recent[field]))
                expected = self.X_train[field].value_counts().reindex(categories, fill_value=0).to_numpy()
                actual = recent[field].value_counts().reindex(categories, fill_value=0).to_numpy()
            p = (expected+.5)/(expected.sum()+.5*len(expected))
            q = (actual+.5)/(actual.sum()+.5*len(actual))
            psi = float(np.sum((q-p)*np.log(q/p)))
            features.append({'feature':field, 'psi':round(psi,4), 'status':'Review' if psi>=.25 else ('Watch' if psi>=.1 else 'Stable')})
        features.sort(key=lambda r:r['psi'], reverse=True)
        return {'status': features[0]['status'], 'samples':len(recent), 'features':features,
            'note':'PSI on up to 200 recent saved applications versus training data. 0.10/0.25 are heuristic alert thresholds, not statistical proof of drift.'}
