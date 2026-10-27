
'use strict';

if (!window.noodleBLE || !window.NoodleAIBLE || !window.NoodleMetrics) {
  throw new Error('NoodleAI runtime did not load. Hard-refresh and verify js/ble.js and js/metrics.js are published.');
}


const $=id=>document.getElementById(id);
const REP_LABEL={accel:'Accelerometer',gyro:'Gyroscope','accel+gyro':'Accel + Gyro',quaternion:'Relative Quaternion',velocity:'Estimated Velocity','velocity+quaternion':'Velocity + Quaternion','accel-dir+mag':'Accel direction + magnitude (unit XYZ, |a|)'};
const MAX_CLASSES=10; // Keep in sync with NoodleAIModel::MAX_CLASSES in the ESP32 firmware.
const state={
  labels:[],samples:[],rawSamples:[],targets:[],rawLengths:[],durationsMs:[],N:100,setupLocked:false,trainingInProgress:false,
  pendingRaw:null,pendingWindow:null,pendingLabelIndex:null,pendingDurationMs:0,pendingGestureEnd:null,recording:null,
  model:null,scaler:null,pkg:null,int8Pkg:null,history:null,report:null,trainedRep:null,deviceMode:'?',
  live:{t:[],accel:[[],[],[]],gyro:[[],[],[]]},
};

function log(msg){const t=new Date().toLocaleTimeString();$('log').textContent+=`[${t}] ${msg}\n`;$('log').scrollTop=$('log').scrollHeight;}
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function pct(v){return `${(100*v).toFixed(1)}%`;}
function score(v){return Number(v).toFixed(3);}
function setBootStatus(text,active=false){const el=$('bootStatus');if(el)el.textContent=text;const box=el?.closest('.boot-reading');if(box)box.classList.toggle('active',!!active);}
function switchTab(name){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.tab-page').forEach(p=>p.classList.toggle('active',p.id===`tab-${name}`));}

document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>switchTab(b.dataset.tab)));

async function initTf(){try{await tf.setBackend('cpu');await tf.ready();$('tfBadge').textContent=`TF.js ${tf.version.tfjs} · ${tf.getBackend()}`;log(`TensorFlow.js ${tf.version.tfjs} ready (${tf.getBackend()}).`);}catch(e){$('tfBadge').textContent='TensorFlow.js unavailable';log(`TensorFlow.js ERROR: ${e.message}`);}}

function datasetObject(){
  const n=state.targets.length;
  return {Xrows:state.samples,y:Int32Array.from(state.targets),labels:[...state.labels],N:state.N,sampleRate:NAI.SAMPLE_RATE_HZ,
    rawRows:state.rawSamples.length===n&&n?state.rawSamples:null,width:state.N*6,rawLengths:[...state.rawLengths],durationsMs:[...state.durationsMs]};
}

function parseHidden(){const h=$('hiddenLayers').value.split(',').map(s=>s.trim()).filter(Boolean).map(Number);if(!h.length||h.some(v=>!Number.isInteger(v)||v<1||v>512))throw new Error('Hidden layers must be comma-separated integers from 1 to 512.');return h;}

function selectedRep(){return $('representation').value;}
function updateInputAndTopology(){
  state.N=Math.max(5,Math.min(500,Number($('normalizedLength').value)||100));
  const rep=selectedRep(),D=state.N*NAI.REP[rep].channels;
  $('inputDimSummary').textContent=`${state.N} × ${NAI.REP[rep].channels} = ${D}`;
  try{$('topology').textContent=`Topology: ${[D,...parseHidden(),Math.max(state.labels.length,0)].join(' → ')}  [${REP_LABEL[rep]}]`;}catch(e){$('topology').textContent=`Topology: ${e.message}`;}
}

function refreshLabels(preferredRecordIndex=null){
  // Preserve the active capture class when editing labels, or select a newly added class.
  const previousRecordIndex=$('recordLabel').selectedIndex;
  $('labelList').innerHTML='';$('recordLabel').innerHTML='';
  state.labels.forEach((label,i)=>{const a=document.createElement('option');a.value=String(i);a.textContent=label;$('labelList').appendChild(a);const b=a.cloneNode(true);$('recordLabel').appendChild(b);});
  const recordIndex=preferredRecordIndex===null?previousRecordIndex:preferredRecordIndex;
  if(recordIndex>=0&&recordIndex<state.labels.length){
    $('recordLabel').selectedIndex=recordIndex;
    if(preferredRecordIndex!==null)$('labelList').selectedIndex=recordIndex;
  }
  $('classCount').textContent=String(state.labels.length);
  updateInputAndTopology();
}

function refreshDataset(){
  const n=state.targets.length;$('sampleCount').textContent=String(n);$('saveDatasetBtn').disabled=!n;
  $('rawDatasetStatus').textContent=n===0?'—':state.rawSamples.length===n?'available':'legacy only';
  const counts=Array(state.labels.length).fill(0);state.targets.forEach(y=>{if(y>=0&&y<counts.length)counts[y]++;});
  $('classCounts').innerHTML='';counts.forEach((c,i)=>{const s=document.createElement('span');s.className='class-pill';s.textContent=`${state.labels[i]}: ${c}`;$('classCounts').appendChild(s);});
  $('trainBtn').disabled=!(state.setupLocked&&state.labels.length>=2&&n>0)||state.trainingInProgress;
}

function invalidateModel(){if(state.model){try{state.model.dispose();}catch(_){}}state.model=null;state.scaler=null;state.pkg=null;state.int8Pkg=null;state.history=null;state.report=null;state.trainedRep=null;$('saveModelBtn').disabled=true;$('saveInt8Btn').disabled=true;$('quantResult').textContent='Train a model to evaluate INT8 quantization.';$('deployBtn').disabled=true;$('trainResult').textContent='';$('curveSummary').textContent='Train a model to see loss, accuracy, and validation metrics.';renderMetrics();drawTrainingCurves();}

function lockUi(locked){
  state.setupLocked=locked;
  // Gesture length is fixed once capture begins; class definitions are still editable.
  // Never mutate training data/labels while an asynchronous model.fit is running.
  const busy=state.trainingInProgress;
  $('normalizedLength').disabled=locked||busy;
  $('labelEntry').disabled=busy;
  $('addLabelBtn').disabled=busy||state.labels.length>=MAX_CLASSES;
  $('removeLabelBtn').disabled=busy;
  $('lockSetupBtn').disabled=locked||busy;
  $('resetDatasetBtn').disabled=busy;
  $('loadDatasetBtn').disabled=busy;
  $('recordLabel').disabled=!locked||busy;
  $('armRecordBtn').disabled=!locked||busy||!window.noodleBLE.connected;
  $('trainBtn').disabled=!locked||busy||state.labels.length<2||!state.targets.length;
  $('downloadReportBtn').disabled=busy||!state.report;
}

function classEditAllowed(){
  if(state.trainingInProgress){alert('Wait for training to finish before editing classes.');return false;}
  if(state.recording||state.pendingRaw||state.pendingWindow){alert('Save or discard the current gesture before editing classes.');return false;}
  return true;
}

function notifyClassChange(action){
  const hadModel=!!(state.model||state.pkg);
  if(hadModel)invalidateModel();
  refreshDataset();
  lockUi(state.setupLocked);
  if(hadModel){
    $('trainResult').textContent='Classes changed — train again before deploying.';
    log(`${action} Remaining samples preserved; previous trained model invalidated. Retrain to update the output classes.`);
  }else log(action);
}

function resetDataset(confirmFirst=true){
  if(confirmFirst&&(state.targets.length||state.setupLocked)&&!confirm('Clear all recorded samples and unlock the dataset setup?'))return;
  state.labels=[];state.samples=[];state.rawSamples=[];state.targets=[];state.rawLengths=[];state.durationsMs=[];state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.pendingDurationMs=0;state.recording=null;state.pendingGestureEnd=null;state.N=Number($('normalizedLength').value)||100;
  invalidateModel();lockUi(false);refreshLabels();refreshDataset();$('recordProgress').textContent='Define labels and lock setup first.';$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;
}

$('addLabelBtn').addEventListener('click',()=>{
  if(!classEditAllowed())return;
  const label=$('labelEntry').value.trim();if(!label)return;
  if(state.labels.includes(label)){alert('That label already exists.');return;}
  if(state.labels.length>=MAX_CLASSES){alert(`The ESP32 firmware supports at most ${MAX_CLASSES} classes.`);return;}
  if(new TextEncoder().encode(label).length>31){alert('Class labels must be at most 31 UTF-8 bytes for the ESP32 firmware.');return;}
  state.labels.push(label);$('labelEntry').value='';
  refreshLabels(state.labels.length-1);
  notifyClassChange(`Added class “${label}” (${state.labels.length}/${MAX_CLASSES}).`);
});
$('labelEntry').addEventListener('keydown',e=>{if(e.key==='Enter'){$('addLabelBtn').click();e.preventDefault();}});
// Remove a class together with all of its recorded gestures. The parallel arrays
// are filtered by the SAME original row indices to keep .npz export aligned.
function removeClassAndRecordings(classIndex){
  if(classIndex<0||classIndex>=state.labels.length)return 0;
  const n=state.targets.length;
  if(state.samples.length!==n||state.rawLengths.length!==n||state.durationsMs.length!==n||
     (state.rawSamples.length!==0&&state.rawSamples.length!==n)){
    throw new Error('Dataset arrays are inconsistent; class deletion was cancelled to protect the recordings.');
  }
  const retained=[];
  for(let row=0;row<n;row++)if(state.targets[row]!==classIndex)retained.push(row);
  const deleted=n-retained.length;
  state.samples=retained.map(row=>state.samples[row]);
  if(state.rawSamples.length)state.rawSamples=retained.map(row=>state.rawSamples[row]);
  state.rawLengths=retained.map(row=>state.rawLengths[row]);
  state.durationsMs=retained.map(row=>state.durationsMs[row]);
  state.targets=retained.map(row=>state.targets[row]>classIndex?state.targets[row]-1:state.targets[row]);
  state.labels.splice(classIndex,1);
  return deleted;
}

$('removeLabelBtn').addEventListener('click',()=>{
  if(!classEditAllowed())return;
  const i=$('labelList').selectedIndex;if(i<0)return;
  const label=state.labels[i];
  const n=state.targets.filter(target=>target===i).length;
  if(n>0&&!confirm(`Remove class “${label}” and permanently delete its ${n} recorded sample(s)? Other classes and recordings will be kept.`))return;
  try{
    const deleted=removeClassAndRecordings(i);
    const nextRecordIndex=Math.min(i,state.labels.length-1);
    refreshLabels(nextRecordIndex);
    notifyClassChange(`Removed class “${label}” and ${deleted} associated recording(s). ${state.targets.length} recording(s) remain.`);
    $('recordProgress').textContent=state.labels.length
      ?'Class removed. Select a remaining class to continue recording.'
      :'All classes removed. Add a class to continue.';
  }catch(err){alert(err.message);log(`Class removal ERROR: ${err.message}`);}
});
$('normalizedLength').addEventListener('input',updateInputAndTopology);$('representation').addEventListener('change',updateInputAndTopology);$('hiddenLayers').addEventListener('input',updateInputAndTopology);

$('lockSetupBtn').addEventListener('click',()=>{if(state.trainingInProgress)return;const N=Number($('normalizedLength').value);if(!Number.isInteger(N)||N<5||N>500){alert('Use 5..500 normalized gesture points.');return;}if(state.labels.length<2){alert('Define at least two labels first.');return;}state.N=N;lockUi(true);refreshDataset();$('recordProgress').textContent=`Ready. Select a label, click “Use BOOT to record”, then hold BOOT while drawing.`;log(`Dataset locked: ${N} normalized points; ${state.labels.length} labels.`);});
$('resetDatasetBtn').addEventListener('click',()=>resetDataset(true));

$('armRecordBtn').addEventListener('click',async()=>{try{if(!state.setupLocked)throw new Error('Lock the dataset setup first.');if(!window.noodleBLE.connected)throw new Error('Connect the device first.');if($('recordLabel').selectedIndex<0)throw new Error('Choose a label.');await window.noodleBLE.setTraining();$('recordProgress').textContent=`Ready for “${state.labels[$('recordLabel').selectedIndex]}”: hold BOOT, draw, release BOOT.`;}catch(e){alert(e.message);}});

function beginGesture(){
  if(state.deviceMode!=='T'||!state.setupLocked||state.trainingInProgress)return;
  const idx=$('recordLabel').selectedIndex;if(idx<0||idx>=state.labels.length){log('Gesture started but no valid label is selected.');return;}
  state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=idx;state.pendingDurationMs=0;state.pendingGestureEnd=null;state.recording=[];
  $('recordProgress').textContent=`Recording “${state.labels[idx]}” while BOOT is held…`;$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=false;
}

function finishGestureIfReady(){
  if(!state.recording||!state.pendingGestureEnd)return;
  const {count,durationMs}=state.pendingGestureEnd;
  if(state.recording.length<count){$('recordProgress').textContent=`BOOT released; waiting for final BLE samples (${state.recording.length}/${count})…`;return;}
  const raw=new Float32Array(count*6);for(let r=0;r<count;r++)for(let c=0;c<6;c++)raw[r*6+c]=state.recording[r][c];
  state.recording=null;state.pendingGestureEnd=null;
  if(count<2){$('recordProgress').textContent='Gesture too short. Try again.';state.pendingLabelIndex=null;$('discardSampleBtn').disabled=true;return;}
  const rawObj={data:raw,length:count};state.pendingRaw=rawObj;state.pendingWindow=NAI.normalizeRawSixAxis(rawObj,state.N);state.pendingDurationMs=durationMs;
  $('recordProgress').textContent=`Captured ${count} raw samples (${(durationMs/1000).toFixed(2)} s) → normalized to ${state.N} points. Save or discard.`;$('saveSampleBtn').disabled=false;$('discardSampleBtn').disabled=false;
}

$('saveSampleBtn').addEventListener('click',()=>{if(state.trainingInProgress||!state.pendingRaw||!state.pendingWindow||state.pendingLabelIndex==null)return;state.samples.push(Float32Array.from(state.pendingWindow));state.rawSamples.push({data:Float32Array.from(state.pendingRaw.data),length:state.pendingRaw.length});state.targets.push(state.pendingLabelIndex);state.rawLengths.push(state.pendingRaw.length);state.durationsMs.push(state.pendingDurationMs||0);log(`Saved “${state.labels[state.pendingLabelIndex]}”: raw=${state.pendingRaw.length}, normalized=${state.N}×6.`);state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.pendingDurationMs=0;$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;$('recordProgress').textContent='Saved. Select a label and hold BOOT for another gesture.';invalidateModel();refreshDataset();});
$('discardSampleBtn').addEventListener('click',()=>{state.recording=null;state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.pendingGestureEnd=null;$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;$('recordProgress').textContent='Discarded. Hold BOOT when ready for another gesture.';});

$('saveDatasetBtn').addEventListener('click',async()=>{try{const blob=await NAI.buildDatasetNpzBlob(datasetObject());downloadBlob(blob,'noodleai_dataset.npz');log('Saved NAI4 dataset (.npz).');}catch(e){alert(e.message);}});
$('loadDatasetBtn').addEventListener('click',()=>$('datasetFile').click());
$('datasetFile').addEventListener('change',async e=>{try{const file=e.target.files[0];if(!file)return;if((state.targets.length||state.setupLocked)&&!confirm('Replace the current dataset/setup?'))return;const arrays=await NAI.loadNpz(file);const ds=NAI.parseDatasetArrays(arrays);resetDataset(false);state.N=ds.N;$('normalizedLength').value=String(ds.N);state.labels=[...ds.labels];state.samples=ds.Xrows.map(r=>Float32Array.from(r));state.targets=Array.from(ds.y,Number);state.rawSamples=ds.rawRows?ds.rawRows.map(r=>({data:Float32Array.from(r.data),length:r.length})):[];state.rawLengths=arrays.raw_lengths?Array.from(arrays.raw_lengths.data,Number):Array(state.targets.length).fill(ds.N);state.durationsMs=arrays.durations_ms?Array.from(arrays.durations_ms.data,Number):Array(state.targets.length).fill(0);refreshLabels();lockUi(true);refreshDataset();$('recordProgress').textContent=`Loaded ${state.targets.length} samples from ${file.name}.`;log(`Loaded ${file.name}: ${state.targets.length} samples; raw gestures ${state.rawSamples.length===state.targets.length?'available':'not available'}.`);}catch(err){alert(err.message);log(`Dataset load ERROR: ${err.message}`);}finally{e.target.value='';}});

// Results remain explicitly tied to the latest trained model and dataset.
function addCell(tr,tag,value,extraClass=''){
  const e=document.createElement(tag);e.textContent=String(value);
  if(extraClass)e.className=extraClass;
  tr.appendChild(e);return e;
}
function renderMetrics(){
  const report=state.report;
  $('metricsArea').hidden=!report;
  $('downloadReportBtn').disabled=!report||state.trainingInProgress;
  if(!report)return;
  const {train,validation,metadata}=report;
  $('metricTrainAccuracy').textContent=pct(train.accuracy);
  $('metricTrainCount').textContent=`${train.correct} / ${train.n} correct`;
  $('metricValAccuracy').textContent=pct(validation.accuracy);
  $('metricValCount').textContent=`${validation.correct} / ${validation.n} correct`;
  $('metricPrecision').textContent=score(validation.macro.precision);
  $('metricRecall').textContent=score(validation.macro.recall);
  $('metricF1').textContent=score(validation.macro.f1);
  $('metricWeightedF1').textContent=score(validation.weighted.f1);
  const smallest=Math.min(...validation.perClass.map(r=>r.support));
  const caveat=smallest<5?` Caution: the smallest validation class has only ${smallest} example(s), making per-class scores unstable.`:'';
  $('methodologyNote').textContent=`${metadata.method} · seed ${metadata.seed} · ${train.n} train / ${validation.n} validation gestures · ${metadata.epochs} epochs · macro: unweighted class average · precision/recall/F1 shown on a 0–1 scale.${caveat}`;
  const classBody=$('classMetricsBody');classBody.innerHTML='';
  for(const r of validation.perClass){
    const tr=document.createElement('tr');
    [r.label,r.support,score(r.precision),score(r.recall),score(r.f1)].forEach((v,i)=>addCell(tr,'td',v,i===0?'class-name':''));
    classBody.appendChild(tr);
  }
  const head=$('confusionHead');head.innerHTML='';
  const hrow=document.createElement('tr');addCell(hrow,'th','Actual ↓ / Predicted →');
  validation.labels.forEach(label=>addCell(hrow,'th',label));head.appendChild(hrow);
  const body=$('confusionBody');body.innerHTML='';
  validation.matrix.forEach((row,i)=>{
    const tr=document.createElement('tr');addCell(tr,'th',validation.labels[i]);
    row.forEach((n,j)=>addCell(tr,'td',n,n?(i===j?'confusion-hit':'confusion-error'):''));
    body.appendChild(tr);
  });
  // A separate, explicitly labeled INT8 validation panel is essential:
  // the training-curve and FP32 confusion matrix are not quantized results.
  const quant=report.int8Validation;
  $('int8MetricsArea').hidden=!quant;
  if(quant){
    $('int8ValAccuracy').textContent=pct(quant.accuracy);
    $('int8ValCount').textContent=`${quant.correct} / ${quant.n} correct`;
    $('int8Precision').textContent=score(quant.macro.precision);
    $('int8Recall').textContent=score(quant.macro.recall);
    $('int8F1').textContent=score(quant.macro.f1);
    $('int8WeightedF1').textContent=score(quant.weighted.f1);
    const matches=report.fp32Int8Agreement;
    $('int8Agreement').textContent=matches?`${matches.matches} / ${matches.n} (${pct(matches.rate)})`:'—';
    $('int8ClassMetricsBody').innerHTML='';
    for(const r of quant.perClass){
      const tr=document.createElement('tr');
      [r.label,r.support,score(r.precision),score(r.recall),score(r.f1)].forEach((v,i)=>addCell(tr,'td',v,i===0?'class-name':''));
      $('int8ClassMetricsBody').appendChild(tr);
    }
    $('int8ConfusionHead').innerHTML='';
    const hrow=document.createElement('tr');addCell(hrow,'th','Actual ↓ / Predicted →');
    quant.labels.forEach(label=>addCell(hrow,'th',label));
    $('int8ConfusionHead').appendChild(hrow);
    $('int8ConfusionBody').innerHTML='';
    quant.matrix.forEach((row,i)=>{
      const tr=document.createElement('tr');addCell(tr,'th',quant.labels[i]);
      row.forEach((n,j)=>addCell(tr,'td',n,n?(i===j?'confusion-hit':'confusion-error'):''));
      $('int8ConfusionBody').appendChild(tr);
    });
  }
}

$('downloadReportBtn').addEventListener('click',async()=>{
  if(!state.report||state.trainingInProgress)return;
  try{
    $('downloadReportBtn').disabled=true;
    const zip=new JSZip();
    const files=window.NoodleMetrics.buildReportFiles(state.report);
    Object.entries(files).forEach(([name,content])=>zip.file(name,content));
    // Export curves at their actual pixel resolution for paper figures.
    for(const [id,name] of [['lossCanvas','loss_vs_epoch.png'],['accuracyCanvas','accuracy_vs_epoch.png']]){
      const canvas=$(id);
      if(typeof canvas.toDataURL==='function'){
        const data=canvas.toDataURL('image/png');
        if(data.startsWith('data:image/png;base64,'))zip.file(name,data.split(',')[1],{base64:true});
      }
    }
    const blob=await zip.generateAsync({type:'blob'});
    downloadBlob(blob,'noodleai_paper_results.zip');
    log('Exported final-epoch validation report (CSV, metadata, and charts).');
  }catch(e){alert(`Report export failed: ${e.message}`);log(`Report ERROR: ${e.message}`);}
  finally{$('downloadReportBtn').disabled=!state.report||state.trainingInProgress;}
});

async function trainModel(){
  if(state.trainingInProgress)return;
  let model=null,tx=null,vx=null,ty=null,vy=null;
  try{
    if(!state.setupLocked||!state.targets.length)throw new Error('Create or load a dataset first.');
    const hidden=parseHidden(),epochs=Number($('epochs').value),rep=selectedRep();
    if(!Number.isInteger(epochs)||epochs<10||epochs>5000)throw new Error('Epochs must be 10..5000.');
    const ds=datasetObject(),K=ds.labels.length;
    const counts=Array(K).fill(0);ds.y.forEach(y=>{if(y<0||y>=K)throw new Error('Invalid dataset label.');counts[y]++;});
    if(Math.min(...counts)<2)throw new Error('Each class needs at least two samples for a stratified train/validation split.');
    state.trainingInProgress=true;
    // New run supersedes stale metrics and previously trained parameters.
    invalidateModel();lockUi(state.setupLocked);
    $('trainResult').textContent='Preparing representation…';
    if(Math.abs(ds.sampleRate-NAI.SAMPLE_RATE_HZ)>1e-3) throw new Error('LPF5 sample rate must match the 50-Hz NAI4 model sample rate');
    const rows=NAI.buildRepresentationDataset(ds,rep),D=rows[0].length;
    if(D>1536)throw new Error('Input dimension '+D+' exceeds firmware MAX_DIM=1536. Reduce normalized length.');
    const initialSplit=NAI.sklearnStratifiedSplit(ds.y,K,42);
    const split=window.NoodleMetrics.ensureClassCoverage(initialSplit,ds.y,K);
    const scaler=NAI.configureRepresentationScaler(NAI.fitScaler(rows,split.train),rep,ds.N);
    const Xtr=NAI.standardizeRows(rows,split.train,scaler),Xva=NAI.standardizeRows(rows,split.test,scaler);
    const ytr=Int32Array.from(split.train.map(i=>ds.y[i])),yva=Int32Array.from(split.test.map(i=>ds.y[i]));
    model=NAI.makeModel(D,hidden,K,split.train.length);
    tx=tf.tensor2d(NAI.flattenRows(Xtr),[Xtr.length,D],'float32');
    vx=tf.tensor2d(NAI.flattenRows(Xva),[Xva.length,D],'float32');
    if(K===2){
      ty=tf.tensor2d(Float32Array.from(ytr),[ytr.length,1]);
      vy=tf.tensor2d(Float32Array.from(yva),[yva.length,1]);
    }else{
      const ity=tf.tensor1d(ytr,'int32'),ivy=tf.tensor1d(yva,'int32');
      try{ty=tf.oneHot(ity,K);vy=tf.oneHot(ivy,K);}finally{ity.dispose();ivy.dispose();}
    }
    const hist={epoch:[],loss:[],valLoss:[],acc:[],valAcc:[]};
    $('trainResult').textContent=`Training ${D} → ${hidden.join(' → ')} → ${K}…`;
    switchTab('curves');
    await model.fit(tx,ty,{epochs,batchSize:Math.min(200,split.train.length),shuffle:true,
      validationData:[vx,vy],verbose:0,callbacks:{onEpochEnd:async(epoch,l)=>{
        const acc=l.acc??l.accuracy??0,va=l.val_acc??l.val_accuracy??0;
        hist.epoch.push(epoch+1);hist.loss.push(l.loss);hist.valLoss.push(l.val_loss);
        hist.acc.push(acc);hist.valAcc.push(va);
        if(epoch===0||(epoch+1)%5===0||epoch+1===epochs){
          $('curveSummary').textContent=`Epoch ${epoch+1}/${epochs} · loss ${l.loss.toFixed(4)} · validation accuracy ${pct(va)}`;
          drawTrainingCurves(hist);await tf.nextFrame();
        }
      }}});
    const tr=await NAI.predictTf(model,Xtr,K),va=await NAI.predictTf(model,Xva,K);
    const train=window.NoodleMetrics.evaluate(ytr,tr.pred,ds.labels);
    const validation=window.NoodleMetrics.evaluate(yva,va.pred,ds.labels);
    const pkg=await NAI.exportNai4(model,scaler,ds.labels,ds.N,rep);
    // PTQ uses ONLY the training split for calibration. Validation data is
    // never used to derive weight/activation scales or the fixed threshold.
    const int8Pkg=await NAI.exportNai4Int8(model,scaler,ds.labels,ds.N,rep,Xtr,Xva,yva);
    const quantMetrics=window.NoodleMetrics.evaluate(yva,int8Pkg.validationPredictions,ds.labels);
    const int8ValidationPredictions=split.test.map((datasetIndex,i)=>({datasetIndex,actual:Number(yva[i]),predicted:Number(int8Pkg.validationPredictions[i]),probabilities:Array.from(int8Pkg.validationProbabilities[i])}));
    const matches=Array.from(va.pred).reduce((n,p,i)=>n+(Number(p)===Number(int8Pkg.validationPredictions[i])?1:0),0);
    const fp32Int8Agreement={matches,n:yva.length,rate:matches/yva.length};
    const report={train,validation,int8Validation:quantMetrics,fp32Int8Agreement,int8ValidationPredictions,history:hist,split,
      validationPredictions:split.test.map((datasetIndex,i)=>({datasetIndex,actual:Number(yva[i]),predicted:Number(va.pred[i]),probabilities:Array.from(va.probs[i])})),
      metadata:{createdUtc:new Date().toISOString(),model_format:'NAI4 FP32 + NAI4 packed INT8',architecture:pkg.dims,
        representation:rep,representation_label:REP_LABEL[rep],preprocessing:'LPF5-before-resampling',lowpassHz:5.0,centering:rep==='accel'||rep==='gyro'||rep==='accel+gyro'?'per-gesture mean':'none',normalizedLength:ds.N,sampleRateHz:ds.sampleRate,
        labels:[...ds.labels],targets:Array.from(ds.y),hiddenLayers:hidden,epochs,
        optimizer:'Adam',learningRate:0.001,batchSize:Math.min(200,split.train.length),seed:42,
        method:'20% stratified validation holdout (at least one example per class per split)',coverageAdjusted:split.coverageAdjusted,
        testSet:'None; these are held-out validation scores, not external test scores',
        note:'Scaler fitted on training data only. Final epoch evaluated. Repeated validation-guided model selection biases scores.'}};
    state.model=model;model=null;
    state.scaler=scaler;state.pkg=pkg;state.int8Pkg=int8Pkg;state.history=hist;state.report=report;state.trainedRep=rep;
    $('trainResult').textContent=`Train ${pct(train.accuracy)} · validation ${pct(validation.accuracy)} · macro F1 ${score(validation.macro.f1)} · NAI4 ${(pkg.total/1024).toFixed(1)} KiB`;
    $('curveSummary').textContent=`Final epoch ${epochs} · ${REP_LABEL[rep]} · ${train.n} train / ${validation.n} validation samples · seed 42`;
    $('saveModelBtn').disabled=false;$('saveInt8Btn').disabled=false;$('deployBtn').disabled=!window.noodleBLE.connected;
    $('quantResult').textContent=`PTQ INT8 validation: accuracy ${pct(quantMetrics.accuracy)}, macro F1 ${score(quantMetrics.macro.f1)} · agreement with FP32 ${pct(fp32Int8Agreement.rate)} · INT8 raw ${(int8Pkg.total/1024).toFixed(1)} KiB. Separate FP32 and INT8 confusion matrices are on the Metrics tab. Deploy only to INT8 firmware.`;
    drawTrainingCurves(hist);renderMetrics();
    log(`Training complete: ${REP_LABEL[rep]}, topology ${pkg.dims.join('→')}, validation accuracy=${pct(validation.accuracy)}, precision=${score(validation.macro.precision)}, recall=${score(validation.macro.recall)}, macro F1=${score(validation.macro.f1)}.`);
    switchTab('curves');
  }catch(e){alert(e.message);log(`Training ERROR: ${e.message}`);}
  finally{
    for(const t of [tx,vx,ty,vy])if(t)try{t.dispose();}catch(_){}
    if(model)try{model.dispose();}catch(_){}
    state.trainingInProgress=false;lockUi(state.setupLocked);
  }
}
$('trainBtn').addEventListener('click',trainModel);

$('saveModelBtn').addEventListener('click',()=>{if(!state.pkg)return;downloadBlob(state.pkg.blob,`noodleai_${state.trainedRep||'model'}.nai`);});
$('saveInt8Btn').addEventListener('click',()=>{if(!state.int8Pkg)return;downloadBlob(state.int8Pkg.blob,`noodleai_${state.trainedRep||'model'}_int8.nai`);});
$('deployBtn').addEventListener('click',async()=>{if(!state.pkg)return;try{
  const precision=$('deployPrecision').value;
  const devicePrecision=window.noodleBLE.precision;
  if(precision==='int8' && devicePrecision!=='int8') throw new Error('INT8 deployment requires the new INT8 firmware (NAI_PRECISION:INT8). Connected firmware: '+(devicePrecision||'unknown/legacy'));
  if(precision==='fp32' && devicePrecision==='int8') throw new Error('The connected firmware is INT8-only. Select INT8 or flash FP32 firmware.');
  const deployPkg=precision==='int8'?state.int8Pkg:state.pkg;
  if(!deployPkg) throw new Error('Train a model for '+precision+' before deploying.');
  $('deployBtn').disabled=true;$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;const chunk=Number($('chunkSize').value);await window.noodleBLE.deployPackage(deployPkg.files,{chunkSize:chunk,onProgress:p=>{const prog=$('deployProgress');prog.max=Math.max(1,p.total);prog.value=p.sent;const percent=p.total?Math.round(100*p.sent/p.total):0;const msg={begin:'Starting transactional deployment…','file-begin':`Preparing ${p.file}…`,sending:`${p.file}: ${percent}% total`,commit:'Files verified; validating Noodle model…',done:'MODEL_OK — model activated and ready.',error:`Deployment failed: ${p.error||'unknown error'}`}[p.stage]||p.stage;$('deployText').textContent=msg;}});log(`Deployment complete: MODEL_OK (${precision}).`);}catch(e){alert(e.message);log(`Deployment ERROR: ${e.message}`);}finally{$('deployBtn').disabled=!(window.noodleBLE.connected&&state.pkg);$('trainingModeBtn').disabled=!window.noodleBLE.connected;$('inferenceModeBtn').disabled=!window.noodleBLE.connected;}});
$('trainingModeBtn').addEventListener('click',async()=>{try{await window.noodleBLE.setTraining();}catch(e){alert(e.message);}});$('inferenceModeBtn').addEventListener('click',async()=>{try{await window.noodleBLE.setInference();}catch(e){alert(e.message);}});

$('connectBtn').addEventListener('click',async()=>{try{if(window.noodleBLE.connected)await window.noodleBLE.disconnect();else await window.noodleBLE.connect();}catch(e){alert(e.message);log(`BLE ERROR: ${e.message}`);}});
window.noodleBLE.addEventListener('connected',e=>{$('connectBtn').textContent='Disconnect';$('bleBadge').textContent='Connected';$('bleBadge').classList.remove('badge-muted');$('deviceStatus').textContent=`Connected: ${e.detail.name}`;$('supportNote').textContent='Raw six-axis stream active.';$('trainingModeBtn').disabled=false;$('inferenceModeBtn').disabled=false;$('armRecordBtn').disabled=!state.setupLocked||state.trainingInProgress;$('deployBtn').disabled=!state.pkg;setBootStatus('BOOT ready · press to capture');log(`BLE connected to ${e.detail.name}.`);});
window.noodleBLE.addEventListener('disconnected',()=>{$('connectBtn').textContent='Connect';$('bleBadge').textContent='Disconnected';$('bleBadge').classList.add('badge-muted');$('deviceStatus').textContent='Disconnected';$('supportNote').textContent=window.NoodleAIBLE.supportMessage();$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;$('armRecordBtn').disabled=true;$('deployBtn').disabled=true;setBootStatus('Ready when connected');log('BLE disconnected.');});
window.noodleBLE.addEventListener('warning',e=>log(`BLE warning: ${e.detail.text}`));window.noodleBLE.addEventListener('deploy-log',e=>log(e.detail.text));

window.noodleBLE.addEventListener('imu',e=>{const s=e.detail.sample;$('accelStatus').textContent=`ax ${s.ax.toFixed(3)} g   ay ${s.ay.toFixed(3)} g   az ${s.az.toFixed(3)} g`;$('gyroStatus').textContent=`gx ${s.gx.toFixed(1)} °/s   gy ${s.gy.toFixed(1)} °/s   gz ${s.gz.toFixed(1)} °/s`;pushLive(s);if(state.recording){state.recording.push([s.ax,s.ay,s.az,s.gx,s.gy,s.gz]);$('recordProgress').textContent=`Recording: ${state.recording.length} raw samples…`;finishGestureIfReady();}});

window.noodleBLE.addEventListener('status',e=>{const text=e.detail.text;if(e.detail.notify)log(`Device: ${text}`);if(text==='MODE:T'){state.deviceMode='T';$('modeStatus').textContent='Current mode: TRAINING';$('deviceStatus').textContent='Training mode';setBootStatus('BOOT ready · training');}else if(text==='MODE:I'){state.deviceMode='I';$('modeStatus').textContent='Current mode: INFERENCE';$('deviceStatus').textContent='Inference mode';setBootStatus('BOOT ready · inference');}else if(text==='GESTURE:START'){setBootStatus('BOOT held · capturing',true);if(state.deviceMode==='T')beginGesture();else{$('predictionMeta').textContent='Recording gesture…';$('deviceStatus').textContent='Inference: recording gesture…';}}else if(text.startsWith('GESTURE:END:')){setBootStatus('BOOT released · gesture closed');const p=text.split(':');const count=Number(p[2]||0),durationMs=Number(p[3]||0);if(state.deviceMode==='T'&&state.recording){state.pendingGestureEnd={count,durationMs};finishGestureIfReady();}else{$('predictionMeta').textContent=`raw=${count} samples · duration=${(durationMs/1000).toFixed(2)} s`;$('deviceStatus').textContent='Inference: classifying…';}}else if(text.startsWith('GESTURE:SHORT')){setBootStatus('Gesture too short · try again');state.recording=null;state.pendingGestureEnd=null;state.pendingRaw=null;$('recordProgress').textContent='Gesture too short. Hold BOOT a little longer and try again.';$('predictionMeta').textContent='Gesture too short — try again';$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;}else if(text==='MODEL_OK'){switchTab('deploy');$('deployText').textContent='FFat model verified, activated, and ready.';}else if(text.startsWith('P:')){const p=text.split(':');if(p.length>=3){const idx=Number(p[1]),conf=Number(p[2]);const label=state.labels[idx]??String(idx);$('predictionLabel').textContent=label;$('predictionConfidence').textContent=`Confidence ${pct(conf)}`;$('deviceStatus').textContent=`Inference: ${label} (${pct(conf)})`;setBootStatus('BOOT ready · next gesture');}}});

function pushLive(s){const L=250;state.live.t.push(s.t_ms/1000);const av=[s.ax,s.ay,s.az],gv=[s.gx,s.gy,s.gz];for(let c=0;c<3;c++){state.live.accel[c].push(av[c]);state.live.gyro[c].push(gv[c]);}if(state.live.t.length>L){state.live.t.shift();for(const a of state.live.accel)a.shift();for(const a of state.live.gyro)a.shift();}drawLineChart($('accelCanvas'),state.live.accel,['ax','ay','az']);drawLineChart($('gyroCanvas'),state.live.gyro,['gx','gy','gz']);}
$('clearPlotBtn').addEventListener('click',()=>{state.live={t:[],accel:[[],[],[]],gyro:[[],[],[]]};drawLiveEmpty();});

function drawLineChart(canvas,series,labels,{fixedY=null,xLabel=null,percentY=false}={}){const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,pad={l:52,r:14,t:18,b:xLabel?48:28};ctx.clearRect(0,0,W,H);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,W,H);const all=series.flat().filter(Number.isFinite);let lo=fixedY?fixedY[0]:(all.length?Math.min(...all):-1),hi=fixedY?fixedY[1]:(all.length?Math.max(...all):1);if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1;}if(!fixedY){const p=.12*(hi-lo);lo-=p;hi+=p;}ctx.strokeStyle='#e3e8ef';ctx.lineWidth=1;for(let k=0;k<=4;k++){const y=pad.t+k*(H-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}ctx.fillStyle='#758092';ctx.font='12px system-ui';ctx.textAlign='right';for(let k=0;k<=4;k++){const v=hi-(hi-lo)*k/4;const y=pad.t+(H-pad.t-pad.b)*k/4;ctx.fillText(percentY?`${(100*v).toFixed(0)}%`:v.toFixed(2),pad.l-6,y+4);}if(xLabel&&series.some(a=>a.length)){const n=Math.max(...series.map(a=>a.length));ctx.textAlign='left';ctx.fillText(`${xLabel} 1`,pad.l,H-pad.b+17);ctx.textAlign='right';ctx.fillText(`${xLabel} ${n}`,W-pad.r,H-pad.b+17);}const colors=['#2869df','#00a37a','#d88a14','#8f5bd6'];series.forEach((a,c)=>{if(a.length<2)return;ctx.strokeStyle=colors[c%colors.length];ctx.lineWidth=1.8;ctx.beginPath();for(let i=0;i<a.length;i++){const x=pad.l+i*(W-pad.l-pad.r)/Math.max(1,a.length-1),y=pad.t+(hi-a[i])*(H-pad.t-pad.b)/(hi-lo);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();});ctx.textAlign='left';labels.forEach((l,i)=>{ctx.fillStyle=colors[i%colors.length];ctx.fillRect(pad.l+i*82,H-15,12,3);ctx.fillStyle='#596476';ctx.fillText(l,pad.l+17+i*82,H-10);});}
function drawLiveEmpty(){drawLineChart($('accelCanvas'),[[],[],[]],['ax','ay','az']);drawLineChart($('gyroCanvas'),[[],[],[]],['gx','gy','gz']);}

function drawTrainingCurves(h=state.history){
  if(!h||!h.epoch?.length){
    drawLineChart($('lossCanvas'),[[],[]],['train','validation']);
    drawLineChart($('accuracyCanvas'),[[],[]],['train','validation'],{fixedY:[0,1]});
    $('lossNumbers').textContent='Final losses —';$('accuracyNumbers').textContent='Final accuracies —';return;
  }
  drawLineChart($('lossCanvas'),[h.loss,h.valLoss],['train loss','validation loss'],{xLabel:'Epoch'});
  drawLineChart($('accuracyCanvas'),[h.acc,h.valAcc],['train accuracy','validation accuracy'],{fixedY:[0,1],xLabel:'Epoch',percentY:true});
  const last=h.epoch.length-1;
  $('lossNumbers').textContent=`Epoch ${h.epoch[last]}  ·  train loss ${h.loss[last].toFixed(4)}  ·  validation loss ${h.valLoss[last].toFixed(4)}`;
  $('accuracyNumbers').textContent=`Epoch ${h.epoch[last]}  ·  train ${pct(h.acc[last])}  ·  validation ${pct(h.valAcc[last])}`;
}

$('clearLogBtn').addEventListener('click',()=>$('log').textContent='');
$('supportNote').textContent=window.NoodleAIBLE.supportMessage();
refreshLabels();refreshDataset();lockUi(false);updateInputAndTopology();drawLiveEmpty();drawTrainingCurves();initTf();

// First-install USB flashing lives in a separate page from BLE model deployment.
document.getElementById('openFirmwareTabBtn')?.addEventListener('click', () => switchTab('firmware'));
