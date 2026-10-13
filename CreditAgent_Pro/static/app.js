'use strict';
const $ = selector => document.querySelector(selector);
const esc = value => {const el=document.createElement('span');el.textContent=String(value ?? '');return el.innerHTML;};
const percent = value => value === null || value === undefined ? 'N/A' : Number(value).toFixed(1)+'%';
const label = name => (specs.find(f=>f.name===name)||{}).label || name.replaceAll('_',' ');
let specs=[], current=null, historyRows=[], batchRows=[], summary=null, ready=false;
function notice(text,type=''){const el=$('#notice');el.textContent=text;el.className='notice '+type;el.classList.toggle('hidden',!text);}
async function api(url, options={}) {
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),url==='/api/batch'?180000:60000);
  try {
    const response=await fetch(url,{...options,signal:controller.signal,headers:{'X-CreditAgent':'local-ui',...(options.body instanceof FormData?{}:{'Content-Type':'application/json'}),...options.headers}});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error || 'Request failed.');
    return data;
  } catch(error) {
    if(error.name==='AbortError') throw new Error('This request took too long. It may still finish on the server; check application history before trying again.');
    throw error;
  } finally {clearTimeout(timeout);}
}
const post=(url,data)=>api(url,{method:'POST',body:JSON.stringify(data)});
async function busy(button,task,text='Working…') {
  const original=button.innerHTML; button.disabled=true;button.textContent=text;
  try{await task();}catch(error){notice(error.message,'error');$('#notice').scrollIntoView({block:'nearest'});}
  finally{button.disabled=false;button.innerHTML=original;}
}
function fieldElement(field,prefix='') {
  const wrapper=document.createElement('label'), title=document.createElement('span');title.textContent=field.label;wrapper.append(title);
  const input=document.createElement(field.type==='select'?'select':'input');
  input.id=prefix+field.name;input.name=field.name;input.required=true;
  if(field.type==='select') field.options.forEach(([v,t])=>input.add(new Option(t,v)));
  else {input.type='number';input.min=field.min;input.max=field.max;input.step='1';input.value=field.value;}
  wrapper.htmlFor=input.id;wrapper.append(input);return wrapper;
}
function formValues(form) {
  const data=Object.fromEntries(new FormData(form));
  for(const field of specs.filter(f=>f.type==='number' && f.name in data)){
    const value=Number(data[field.name]);
    if(String(data[field.name]).trim()==='' || !Number.isInteger(value) || value<field.min || value>field.max) throw new Error(`Enter a whole number between ${field.min} and ${field.max} for ${field.label}.`);
    data[field.name]=value;
  }
  return data;
}
function metrics(target, rows){$(target).innerHTML=rows.map(([title,value,sub])=>`<article><span>${esc(title)}</span><b>${esc(value)}</b>${sub?`<small>${esc(sub)}</small>`:''}</article>`).join('');}
function table(target,headers,rows){$(target).innerHTML=rows.length?`<table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>`:'<div class="empty">Nothing here yet. Your results will appear here.</div>';}
function showView(name){
  if(!['assessment','history','fairness','monitoring','batch'].includes(name)) name='assessment';
  document.querySelectorAll('.view').forEach(el=>el.classList.toggle('hidden',el.id!==name+'View'));
  document.querySelectorAll('[data-view]').forEach(el=>el.classList.toggle('active',el.dataset.view===name));
  $('#pageName').textContent=document.querySelector(`[data-view="${name}"]`).textContent.trim();
  if(ready){const action={history:loadHistory,fairness:loadFairness,monitoring:loadMonitoring}[name];if(action) action().catch(e=>notice(e.message,'error'));}
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{location.hash=button.dataset.view;showView(button.dataset.view);window.scrollTo({top:0});}));
window.addEventListener('hashchange',()=>{if(location.hash!=='#formCard')showView(location.hash.slice(1));});

async function waitForEngine(){
  try{
    const health=await api('/api/health');
    if(health.status==='error'){notice('Startup needs attention: '+health.error,'error');$('#healthStatus').textContent='Setup required';return;}
    if(health.status!=='ready'){setTimeout(waitForEngine,2000);return;}
    summary=await api('/api/summary');ready=true;
    metrics('#modelMetrics',[['SELECTED MODEL',summary.model,'3-fold CV selection'],['HELD-OUT ACCURACY',percent(summary.accuracy),summary.test_rows+' test cases'],['BAD-RISK RECALL',percent(summary.recall),'At the selected threshold'],['COST THRESHOLD',summary.threshold.toFixed(2),summary.cost_ratio]]);
    $('#submitBtn').disabled=false;$('#healthStatus').textContent='Models ready';$('#healthStatus').classList.add('ready');notice('');
    showView(location.hash.slice(1));
  }catch(error){notice('Cannot connect: '+error.message,'error');setTimeout(waitForEngine,5000);}
}
async function boot(){
  try{
    specs=await api('/api/schema');$('#fields').replaceChildren(...specs.map(f=>fieldElement(f)));
    $('#whatFields').replaceChildren(...specs.filter(f=>['credit_amount','duration_months','savings_account','installment_rate'].includes(f.name)).map(f=>fieldElement(f,'what_')));
    showView(location.hash.slice(1));await waitForEngine();
  }catch(error){notice(error.message+' Refresh the page after checking the terminal.','error');}
}
$('#creditForm').addEventListener('submit',event=>{
  event.preventDefault();busy($('#submitBtn'),async()=>{
    const input=formValues(event.target),result=await post('/api/assess',input);
    current={input,result,reviews:[]};renderResult();notice('Assessment saved. Review the explanation and record your judgment.','success');
    $('#result').scrollIntoView({behavior:'smooth',block:'start'});
  },'Six agents are working…');
});
function renderResult(){
  const r=current.result;$('#result').classList.remove('hidden');
  $('#assessmentId').textContent=r.assessment_id;$('#riskLabel').textContent=r.risk_label;
  $('#riskBadge').textContent='Human review required';$('#recommendation').textContent=r.recommendation;
  $('#pdfLink').href='/api/report/'+encodeURIComponent(r.assessment_id);
  metrics('#resultMetrics',[['BAD-RISK PROBABILITY',percent(r.bad_probability)],['GOOD-RISK PROBABILITY',percent(r.good_probability)],['PREDICTED-CLASS PROBABILITY',percent(r.confidence)],['ASSESSMENT TIME',r.latency_ms.toFixed(0)+' ms']]);
  const ex=r.explanation;
  if(ex.status==='ready'){
    $('#shapInfo').textContent=`Top 8 of 17 contributions · ${ex.units} · ${ex.background_rows} training reference rows. Baseline: ${ex.baseline.toFixed(3)}.`;
    const max=Math.max(...ex.contributions.map(x=>Math.abs(x.impact)),.00001);
    $('#contributions').innerHTML=ex.contributions.slice(0,8).map(x=>`<div class="impact-row"><div class="impact-label"><span>${esc(label(x.feature))}</span><b>${x.impact>=0?'+':''}${x.impact.toFixed(3)}</b></div><div class="impact-track"><i class="${x.impact<0?'down':''}" style="width:${Math.abs(x.impact)/max*100}%"></i></div></div>`).join('');
  }else{$('#shapInfo').textContent='SHAP unavailable: '+ex.message;$('#contributions').replaceChildren();}
  $('#trace').innerHTML=r.agent_trace.map((name,i)=>`<div><b>0${i+1}</b><span>${esc(name)}</span><span>${name.includes('unavailable')?'!':'✓'}</span></div>`).join('');
  $('#costInfo').textContent=r.cost_status;$('#versionInfo').textContent=r.model_version+' · '+new Date(r.created_at).toLocaleString();
  for(const input of $('#whatForm').elements)if(input.name in current.input)input.value=current.input[input.name];
  $('#whatResult').textContent='';$('#reviewStatus').textContent='';$('#reviewForm').reset();
  if(current.reviewer)$('#reviewForm').elements.reviewer.value=current.reviewer;
  $('#reviewEvents').innerHTML=current.reviews.map(event=>`<div class="review-event"><b>${esc(event.decision)}</b> · ${esc(event.reviewer)}<br>${esc(event.reason)}<br>${esc(new Date(event.created_at).toLocaleString())}</div>`).join('');
}
$('#whatForm').addEventListener('submit',event=>{event.preventDefault();busy($('#whatBtn'),async()=>{
  if(!current)throw new Error('Run an assessment first.');
  const r=await post('/api/what-if',{assessment_id:current.result.assessment_id,changes:formValues(event.target)});
  $('#whatResult').textContent=`Bad risk: ${percent(r.original_bad_probability)} → ${percent(r.new_bad_probability)} (${r.difference>=0?'+':''}${r.difference.toFixed(1)} percentage points). ${r.new_risk_label}. ${r.note}`;
});});
$('#reviewForm').addEventListener('submit',event=>{event.preventDefault();busy($('#reviewBtn'),async()=>{
  if(!current)throw new Error('Run an assessment first.');
  await post('/api/review/'+current.result.assessment_id,Object.fromEntries(new FormData(event.target)));
  current=await api('/api/assessment/'+current.result.assessment_id);renderResult();$('#reviewStatus').textContent='Review saved to the audit trail. The PDF includes your review.';
});});
async function loadHistory(){historyRows=await api('/api/history');renderHistory();}
function renderHistory(){
  const search=$('#historySearch').value.toLowerCase();
  const rows=historyRows.filter(r=>`${r.id} ${r.result.risk_label} ${r.reviewer_decision}`.toLowerCase().includes(search));
  table('#historyTable',['Assessment','Created','Risk','Bad probability','Reviewer action','Open'],rows.map(r=>[esc(r.id),esc(new Date(r.created_at).toLocaleString()),esc(r.result.risk_label),percent(r.result.bad_probability),esc(r.reviewer_decision),`<button data-open="${esc(r.id)}">View →</button>`]));
}
$('#historySearch').addEventListener('input',renderHistory);
$('#refreshHistory').addEventListener('click',()=>busy($('#refreshHistory'),loadHistory));
$('#historyTable').addEventListener('click',event=>{const button=event.target.closest('[data-open]');if(button)busy(button,async()=>{
  current=await api('/api/assessment/'+button.dataset.open);location.hash='assessment';showView('assessment');renderResult();$('#result').scrollIntoView({behavior:'smooth'});
});});
async function loadFairness(){
  const r=await api('/api/fairness');$('#fairnessNote').textContent=r.note;
  table('#fairnessTable',['Attribute','Group','Cases','Accuracy','Bad recall','False positive','False negative','Sample flag'],r.groups.map(g=>[esc(g.attribute.replaceAll('_',' ')),esc(g.group),g.samples,percent(g.accuracy),percent(g.recall),percent(g.fpr),percent(g.fnr),g.small_sample?'<span class="chip">Small sample</span>':'—']));
  table('#gapTable',['Attribute','Accuracy gap','Recall gap','FPR gap','FNR gap'],r.gaps.map(g=>[esc(g.attribute.replaceAll('_',' ')),...['accuracy','recall','fpr','fnr'].map(k=>g[k]===null?'N/A':g[k].toFixed(1)+' pp')]));
}
async function loadMonitoring(){
  const r=await api('/api/monitoring');
  metrics('#monitorMetrics',[['SAVED ASSESSMENTS',r.assessment_count],['AWAITING FIRST REVIEW',r.pending],['HIGHER-RISK SHARE',percent(r.bad_risk_rate)],['MEAN ASSESSMENT TIME',r.average_latency_ms===null?'N/A':r.average_latency_ms.toFixed(0)+' ms']]);
  $('#trackingInfo').innerHTML=`<span class="chip">${esc(r.tracking.status)}</span><p>${esc(r.tracking.message)}</p><p class="code">${esc(r.model_version)}</p><p>Mean predicted-class probability: ${percent(r.average_confidence)}</p>`;
  $('#disagreementInfo').innerHTML=`<div class="big-number">${percent(r.disagreement_rate)}</div><p>${r.agreement_denominator} definite opinions across ${r.reviewed} reviewed applications.</p>`;
  const drift=r.drift;$('#driftBadge').textContent=drift.status;
  $('#driftInfo').textContent=drift.samples<30?`${drift.samples} saved applications for this model. Drift evaluation starts at 30; simulations are excluded.`:drift.note;
  table('#driftTable',['Input field','PSI','Signal'],drift.features.map(f=>[esc(label(f.feature)),f.psi.toFixed(4),esc(f.status)]));
  table('#comparisonTable',['Model','CV ROC-AUC','CV accuracy','CV recall','CV F1'],summary.comparison.map(c=>[esc(c.model),...['auc','accuracy','recall','f1'].map(k=>percent(c[k]))]));
}
$('#refreshMonitoring').addEventListener('click',()=>busy($('#refreshMonitoring'),loadMonitoring));
$('#batchForm').addEventListener('submit',event=>{event.preventDefault();busy($('#batchBtn'),async()=>{
  const file=$('#batchFile').files[0];if(!file)throw new Error('Choose a CSV file.');if(file.size>2*1024*1024)throw new Error('CSV must be below 2 MB.');
  const form=new FormData();form.append('file',file);$('#batchStatus').textContent='Assessing rows. Keep this page open…';
  try{batchRows=await api('/api/batch',{method:'POST',body:form});}catch(e){$('#batchStatus').textContent='Batch request did not complete. Check history before resubmitting.';throw e;}
  const saved=batchRows.filter(r=>!r.error).length;$('#batchStatus').textContent=`${saved} saved · ${batchRows.length-saved} failed. Failed rows show their validation error below.`;
  table('#batchTable',['Row','Assessment','Risk','Bad probability','Error'],batchRows.map(r=>[r.row,esc(r.assessment_id||'—'),esc(r.risk||'—'),r.error?'—':percent(r.bad_probability),esc(r.error||'—')]));$('#exportBatch').classList.remove('hidden');
},'Assessing batch…');});
$('#exportBatch').addEventListener('click',()=>{
  const columns=['row','assessment_id','risk','bad_probability','error'];
  const quote=value=>'"'+String(value??'').replace(/^[=+@-]/,"'").replaceAll('"','""')+'"';
  const text=[columns.join(','),...batchRows.map(r=>columns.map(k=>quote(r[k])).join(','))].join('\r\n');
  const url=URL.createObjectURL(new Blob([text],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='credit_batch_results.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
boot();