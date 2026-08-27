
'use strict';

const $=id=>document.getElementById(id);
const REP_LABEL={accel:'Accelerometer',gyro:'Gyroscope','accel+gyro':'Accel + Gyro',quaternion:'Relative Quaternion',velocity:'Estimated Velocity','velocity+quaternion':'Velocity + Quaternion'};
const state={
  labels:[],samples:[],rawSamples:[],targets:[],rawLengths:[],durationsMs:[],N:100,setupLocked:false,
  pendingRaw:null,pendingWindow:null,pendingLabelIndex:null,pendingDurationMs:0,pendingGestureEnd:null,recording:null,
  model:null,scaler:null,pkg:null,history:null,trainedRep:null,deviceMode:'?',
  live:{t:[],accel:[[],[],[]],gyro:[[],[],[]]},
};

function log(msg){const t=new Date().toLocaleTimeString();$('log').textContent+=`[${t}] ${msg}\n`;$('log').scrollTop=$('log').scrollHeight;}
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function pct(v){return `${(100*v).toFixed(1)}%`;}
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

function refreshLabels(){
  $('labelList').innerHTML='';$('recordLabel').innerHTML='';
  state.labels.forEach((label,i)=>{const a=document.createElement('option');a.value=String(i);a.textContent=label;$('labelList').appendChild(a);const b=a.cloneNode(true);$('recordLabel').appendChild(b);});
  $('classCount').textContent=String(state.labels.length);
  updateInputAndTopology();
}

function refreshDataset(){
  const n=state.targets.length;$('sampleCount').textContent=String(n);$('saveDatasetBtn').disabled=!n;
  $('rawDatasetStatus').textContent=n===0?'—':state.rawSamples.length===n?'available':'legacy only';
  const counts=Array(state.labels.length).fill(0);state.targets.forEach(y=>{if(y>=0&&y<counts.length)counts[y]++;});
  $('classCounts').innerHTML='';counts.forEach((c,i)=>{const s=document.createElement('span');s.className='class-pill';s.textContent=`${state.labels[i]}: ${c}`;$('classCounts').appendChild(s);});
  $('trainBtn').disabled=!(state.setupLocked&&n>0);
}

function invalidateModel(){if(state.model){try{state.model.dispose();}catch(_){}}state.model=null;state.scaler=null;state.pkg=null;state.history=null;state.trainedRep=null;$('saveModelBtn').disabled=true;$('deployBtn').disabled=true;$('trainResult').textContent='';$('curveSummary').textContent='Train a model to see loss and accuracy history.';drawTrainingCurves();}

function lockUi(locked){
  state.setupLocked=locked;$('normalizedLength').disabled=locked;$('labelEntry').disabled=locked;$('addLabelBtn').disabled=locked;$('removeLabelBtn').disabled=locked;$('lockSetupBtn').disabled=locked;
  $('recordLabel').disabled=!locked;$('armRecordBtn').disabled=!(locked&&noodleBLE.connected);$('trainBtn').disabled=!(locked&&state.targets.length);
}

function resetDataset(confirmFirst=true){
  if(confirmFirst&&(state.targets.length||state.setupLocked)&&!confirm('Clear all recorded samples and unlock the dataset setup?'))return;
  state.labels=[];state.samples=[];state.rawSamples=[];state.targets=[];state.rawLengths=[];state.durationsMs=[];state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.recording=null;state.pendingGestureEnd=null;state.N=Number($('normalizedLength').value)||100;
  invalidateModel();lockUi(false);refreshLabels();refreshDataset();$('recordProgress').textContent='Define labels and lock setup first.';$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;
}

$('addLabelBtn').addEventListener('click',()=>{if(state.setupLocked)return;const label=$('labelEntry').value.trim();if(!label)return;if(state.labels.includes(label)){alert('That label already exists.');return;}state.labels.push(label);$('labelEntry').value='';refreshLabels();});
$('labelEntry').addEventListener('keydown',e=>{if(e.key==='Enter'){$('addLabelBtn').click();e.preventDefault();}});
$('removeLabelBtn').addEventListener('click',()=>{if(state.setupLocked)return;const i=$('labelList').selectedIndex;if(i>=0){state.labels.splice(i,1);refreshLabels();}});
$('normalizedLength').addEventListener('input',updateInputAndTopology);$('representation').addEventListener('change',updateInputAndTopology);$('hiddenLayers').addEventListener('input',updateInputAndTopology);

$('lockSetupBtn').addEventListener('click',()=>{const N=Number($('normalizedLength').value);if(!Number.isInteger(N)||N<5||N>500){alert('Use 5..500 normalized gesture points.');return;}if(state.labels.length<2){alert('Define at least two labels first.');return;}state.N=N;lockUi(true);refreshDataset();$('recordProgress').textContent=`Ready. Select a label, click “Use BOOT to record”, then hold BOOT while drawing.`;log(`Dataset locked: ${N} normalized points; ${state.labels.length} labels.`);});
$('resetDatasetBtn').addEventListener('click',()=>resetDataset(true));

$('armRecordBtn').addEventListener('click',async()=>{try{if(!state.setupLocked)throw new Error('Lock the dataset setup first.');if(!noodleBLE.connected)throw new Error('Connect the device first.');if($('recordLabel').selectedIndex<0)throw new Error('Choose a label.');await noodleBLE.setTraining();$('recordProgress').textContent=`Ready for “${state.labels[$('recordLabel').selectedIndex]}”: hold BOOT, draw, release BOOT.`;}catch(e){alert(e.message);}});

function beginGesture(){
  if(state.deviceMode!=='T'||!state.setupLocked)return;
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

$('saveSampleBtn').addEventListener('click',()=>{if(!state.pendingRaw||!state.pendingWindow||state.pendingLabelIndex==null)return;state.samples.push(Float32Array.from(state.pendingWindow));state.rawSamples.push({data:Float32Array.from(state.pendingRaw.data),length:state.pendingRaw.length});state.targets.push(state.pendingLabelIndex);state.rawLengths.push(state.pendingRaw.length);state.durationsMs.push(state.pendingDurationMs||0);log(`Saved “${state.labels[state.pendingLabelIndex]}”: raw=${state.pendingRaw.length}, normalized=${state.N}×6.`);state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.pendingDurationMs=0;$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;$('recordProgress').textContent='Saved. Select a label and hold BOOT for another gesture.';invalidateModel();refreshDataset();});
$('discardSampleBtn').addEventListener('click',()=>{state.recording=null;state.pendingRaw=null;state.pendingWindow=null;state.pendingLabelIndex=null;state.pendingGestureEnd=null;$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;$('recordProgress').textContent='Discarded. Hold BOOT when ready for another gesture.';});

$('saveDatasetBtn').addEventListener('click',async()=>{try{const blob=await NAI.buildDatasetNpzBlob(datasetObject());downloadBlob(blob,'noodleai_dataset.npz');log('Saved NAI4 dataset (.npz).');}catch(e){alert(e.message);}});
$('loadDatasetBtn').addEventListener('click',()=>$('datasetFile').click());
$('datasetFile').addEventListener('change',async e=>{try{const file=e.target.files[0];if(!file)return;if((state.targets.length||state.setupLocked)&&!confirm('Replace the current dataset/setup?'))return;const arrays=await NAI.loadNpz(file);const ds=NAI.parseDatasetArrays(arrays);resetDataset(false);state.N=ds.N;$('normalizedLength').value=String(ds.N);state.labels=[...ds.labels];state.samples=ds.Xrows.map(r=>Float32Array.from(r));state.targets=Array.from(ds.y,Number);state.rawSamples=ds.rawRows?ds.rawRows.map(r=>({data:Float32Array.from(r.data),length:r.length})):[];state.rawLengths=arrays.raw_lengths?Array.from(arrays.raw_lengths.data,Number):Array(state.targets.length).fill(ds.N);state.durationsMs=arrays.durations_ms?Array.from(arrays.durations_ms.data,Number):Array(state.targets.length).fill(0);refreshLabels();lockUi(true);refreshDataset();$('recordProgress').textContent=`Loaded ${state.targets.length} samples from ${file.name}.`;log(`Loaded ${file.name}: ${state.targets.length} samples; raw gestures ${state.rawSamples.length===state.targets.length?'available':'not available'}.`);}catch(err){alert(err.message);log(`Dataset load ERROR: ${err.message}`);}finally{e.target.value='';}});

async function trainModel(){
  try{
    if(!state.setupLocked||!state.targets.length)throw new Error('Create or load a dataset first.');
    const hidden=parseHidden(),epochs=Number($('epochs').value),rep=selectedRep();if(!Number.isInteger(epochs)||epochs<10||epochs>5000)throw new Error('Epochs must be 10..5000.');
    const ds=datasetObject();const K=state.labels.length;const counts=Array(K).fill(0);state.targets.forEach(y=>counts[y]++);if(Math.min(...counts)<2)throw new Error('Each class needs at least two samples for a stratified train/validation split.');
    $('trainBtn').disabled=true;$('trainResult').textContent='Preparing representation…';
    const rows=NAI.buildRepresentationDataset(ds,rep);const D=rows[0].length;const split=NAI.sklearnStratifiedSplit(ds.y,K,42);const scaler=NAI.fitScaler(rows,split.train);const Xtr=NAI.standardizeRows(rows,split.train,scaler),Xva=NAI.standardizeRows(rows,split.test,scaler);const ytr=Int32Array.from(split.train.map(i=>ds.y[i])),yva=Int32Array.from(split.test.map(i=>ds.y[i]));
    const model=NAI.makeModel(D,hidden,K,split.train.length);const tx=tf.tensor2d(NAI.flattenRows(Xtr),[Xtr.length,D],'float32'),vx=tf.tensor2d(NAI.flattenRows(Xva),[Xva.length,D],'float32');let ty,vy;if(K===2){ty=tf.tensor2d(Float32Array.from(ytr),[ytr.length,1]);vy=tf.tensor2d(Float32Array.from(yva),[yva.length,1]);}else{const ity=tf.tensor1d(ytr,'int32'),ivy=tf.tensor1d(yva,'int32');ty=tf.oneHot(ity,K);vy=tf.oneHot(ivy,K);ity.dispose();ivy.dispose();}
    const hist={epoch:[],loss:[],valLoss:[],acc:[],valAcc:[]};$('trainResult').textContent=`Training ${D} → ${hidden.join(' → ')} → ${K}…`;switchTab('curves');
    await model.fit(tx,ty,{epochs,batchSize:Math.min(200,split.train.length),shuffle:true,validationData:[vx,vy],verbose:0,callbacks:{onEpochEnd:async(epoch,l)=>{const acc=l.acc??l.accuracy??0,va=l.val_acc??l.val_accuracy??0;hist.epoch.push(epoch+1);hist.loss.push(l.loss);hist.valLoss.push(l.val_loss);hist.acc.push(acc);hist.valAcc.push(va);if(epoch===0||(epoch+1)%5===0||epoch+1===epochs){$('curveSummary').textContent=`Epoch ${epoch+1}/${epochs} · loss ${l.loss.toFixed(4)} · validation accuracy ${pct(va)}`;drawTrainingCurves(hist);await tf.nextFrame();}}}});
    tx.dispose();vx.dispose();ty.dispose();vy.dispose();
    const tr=await NAI.predictTf(model,Xtr,K),va=await NAI.predictTf(model,Xva,K);const trainAcc=NAI.accuracy(tr.pred,Array.from(ytr)),valAcc=NAI.accuracy(va.pred,Array.from(yva));const pkg=await NAI.exportNai4(model,scaler,state.labels,state.N,rep);
    if(state.model){try{state.model.dispose();}catch(_){}}state.model=model;state.scaler=scaler;state.pkg=pkg;state.history=hist;state.trainedRep=rep;
    $('trainResult').textContent=`Train ${pct(trainAcc)} · validation ${pct(valAcc)} · NAI4 ${(pkg.total/1024).toFixed(1)} KiB`;$('curveSummary').textContent=`Finished ${epochs} epochs · train ${pct(trainAcc)} · validation ${pct(valAcc)} · ${REP_LABEL[rep]}`;$('saveModelBtn').disabled=false;$('deployBtn').disabled=!noodleBLE.connected;drawTrainingCurves(hist);log(`Training complete: ${REP_LABEL[rep]}, topology ${pkg.dims.join('→')}, train=${pct(trainAcc)}, validation=${pct(valAcc)}, NAI4=${(pkg.total/1024).toFixed(1)} KiB.`);switchTab('dataset');
  }catch(e){alert(e.message);log(`Training ERROR: ${e.message}`);}finally{$('trainBtn').disabled=!(state.setupLocked&&state.targets.length);}
}
$('trainBtn').addEventListener('click',trainModel);

$('saveModelBtn').addEventListener('click',()=>{if(!state.pkg)return;downloadBlob(state.pkg.blob,`noodleai_${state.trainedRep||'model'}.nai`);});
$('deployBtn').addEventListener('click',async()=>{if(!state.pkg)return;try{$('deployBtn').disabled=true;$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;const chunk=Number($('chunkSize').value);await noodleBLE.deployPackage(state.pkg.files,{chunkSize:chunk,onProgress:p=>{const prog=$('deployProgress');prog.max=Math.max(1,p.total);prog.value=p.sent;const percent=p.total?Math.round(100*p.sent/p.total):0;const msg={begin:'Starting transactional deployment…','file-begin':`Preparing ${p.file}…`,sending:`${p.file}: ${percent}% total`,commit:'Files verified; validating Noodle model…',done:'MODEL_OK — model activated and ready.',error:`Deployment failed: ${p.error||'unknown error'}`}[p.stage]||p.stage;$('deployText').textContent=msg;}});log('Deployment complete: MODEL_OK.');}catch(e){alert(e.message);log(`Deployment ERROR: ${e.message}`);}finally{$('deployBtn').disabled=!(noodleBLE.connected&&state.pkg);$('trainingModeBtn').disabled=!noodleBLE.connected;$('inferenceModeBtn').disabled=!noodleBLE.connected;}});
$('trainingModeBtn').addEventListener('click',async()=>{try{await noodleBLE.setTraining();}catch(e){alert(e.message);}});$('inferenceModeBtn').addEventListener('click',async()=>{try{await noodleBLE.setInference();}catch(e){alert(e.message);}});

$('connectBtn').addEventListener('click',async()=>{try{if(noodleBLE.connected)await noodleBLE.disconnect();else await noodleBLE.connect();}catch(e){alert(e.message);log(`BLE ERROR: ${e.message}`);}});
noodleBLE.addEventListener('connected',e=>{$('connectBtn').textContent='Disconnect';$('bleBadge').textContent='Connected';$('bleBadge').classList.remove('badge-muted');$('deviceStatus').textContent=`Connected: ${e.detail.name}`;$('supportNote').textContent='Raw six-axis stream active.';$('trainingModeBtn').disabled=false;$('inferenceModeBtn').disabled=false;$('armRecordBtn').disabled=!state.setupLocked;$('deployBtn').disabled=!state.pkg;log(`BLE connected to ${e.detail.name}.`);});
noodleBLE.addEventListener('disconnected',()=>{$('connectBtn').textContent='Connect';$('bleBadge').textContent='Disconnected';$('bleBadge').classList.add('badge-muted');$('deviceStatus').textContent='Disconnected';$('supportNote').textContent=NoodleAIBLE.supportMessage();$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;$('armRecordBtn').disabled=true;$('deployBtn').disabled=true;log('BLE disconnected.');});
noodleBLE.addEventListener('warning',e=>log(`BLE warning: ${e.detail.text}`));noodleBLE.addEventListener('deploy-log',e=>log(e.detail.text));

noodleBLE.addEventListener('imu',e=>{const s=e.detail.sample;$('accelStatus').textContent=`ax ${s.ax.toFixed(3)} g   ay ${s.ay.toFixed(3)} g   az ${s.az.toFixed(3)} g`;$('gyroStatus').textContent=`gx ${s.gx.toFixed(1)} °/s   gy ${s.gy.toFixed(1)} °/s   gz ${s.gz.toFixed(1)} °/s`;pushLive(s);if(state.recording){state.recording.push([s.ax,s.ay,s.az,s.gx,s.gy,s.gz]);$('recordProgress').textContent=`Recording: ${state.recording.length} raw samples…`;finishGestureIfReady();}});

noodleBLE.addEventListener('status',e=>{const text=e.detail.text;if(e.detail.notify)log(`Device: ${text}`);if(text==='MODE:T'){state.deviceMode='T';$('modeStatus').textContent='Current mode: TRAINING';$('deviceStatus').textContent='Training mode';}else if(text==='MODE:I'){state.deviceMode='I';$('modeStatus').textContent='Current mode: INFERENCE';$('deviceStatus').textContent='Inference mode';}else if(text==='GESTURE:START'){if(state.deviceMode==='T')beginGesture();else{$('predictionMeta').textContent='Recording gesture…';$('deviceStatus').textContent='Inference: recording gesture…';}}else if(text.startsWith('GESTURE:END:')){const p=text.split(':');const count=Number(p[2]||0),durationMs=Number(p[3]||0);if(state.deviceMode==='T'&&state.recording){state.pendingGestureEnd={count,durationMs};finishGestureIfReady();}else{$('predictionMeta').textContent=`raw=${count} samples · duration=${(durationMs/1000).toFixed(2)} s`;$('deviceStatus').textContent='Inference: classifying…';}}else if(text.startsWith('GESTURE:SHORT')){state.recording=null;state.pendingGestureEnd=null;state.pendingRaw=null;$('recordProgress').textContent='Gesture too short. Hold BOOT a little longer and try again.';$('predictionMeta').textContent='Gesture too short — try again';$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;}else if(text==='MODEL_OK'){switchTab('deploy');$('deployText').textContent='FFat model verified, activated, and ready.';}else if(text.startsWith('P:')){const p=text.split(':');if(p.length>=3){const idx=Number(p[1]),conf=Number(p[2]);const label=state.labels[idx]??String(idx);$('predictionLabel').textContent=label;$('predictionConfidence').textContent=`Confidence ${pct(conf)}`;$('deviceStatus').textContent=`Inference: ${label} (${pct(conf)})`;}}});

function pushLive(s){const L=250;state.live.t.push(s.t_ms/1000);const av=[s.ax,s.ay,s.az],gv=[s.gx,s.gy,s.gz];for(let c=0;c<3;c++){state.live.accel[c].push(av[c]);state.live.gyro[c].push(gv[c]);}if(state.live.t.length>L){state.live.t.shift();for(const a of state.live.accel)a.shift();for(const a of state.live.gyro)a.shift();}drawLineChart($('accelCanvas'),state.live.accel,['ax','ay','az']);drawLineChart($('gyroCanvas'),state.live.gyro,['gx','gy','gz']);}
$('clearPlotBtn').addEventListener('click',()=>{state.live={t:[],accel:[[],[],[]],gyro:[[],[],[]]};drawLiveEmpty();});

function drawLineChart(canvas,series,labels,{fixedY=null}={}){const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,pad={l:42,r:14,t:18,b:28};ctx.clearRect(0,0,W,H);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,W,H);const all=series.flat().filter(Number.isFinite);let lo=fixedY?fixedY[0]:(all.length?Math.min(...all):-1),hi=fixedY?fixedY[1]:(all.length?Math.max(...all):1);if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1;}if(!fixedY){const p=.12*(hi-lo);lo-=p;hi+=p;}ctx.strokeStyle='#e3e8ef';ctx.lineWidth=1;for(let k=0;k<=4;k++){const y=pad.t+k*(H-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}ctx.fillStyle='#758092';ctx.font='12px system-ui';ctx.textAlign='right';ctx.fillText(hi.toFixed(2),pad.l-6,pad.t+4);ctx.fillText(lo.toFixed(2),pad.l-6,H-pad.b);const colors=['#2869df','#00a37a','#d88a14','#8f5bd6'];series.forEach((a,c)=>{if(a.length<2)return;ctx.strokeStyle=colors[c%colors.length];ctx.lineWidth=1.8;ctx.beginPath();for(let i=0;i<a.length;i++){const x=pad.l+i*(W-pad.l-pad.r)/Math.max(1,a.length-1),y=pad.t+(hi-a[i])*(H-pad.t-pad.b)/(hi-lo);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();});ctx.textAlign='left';labels.forEach((l,i)=>{ctx.fillStyle=colors[i%colors.length];ctx.fillRect(pad.l+i*82,H-15,12,3);ctx.fillStyle='#596476';ctx.fillText(l,pad.l+17+i*82,H-10);});}
function drawLiveEmpty(){drawLineChart($('accelCanvas'),[[],[],[]],['ax','ay','az']);drawLineChart($('gyroCanvas'),[[],[],[]],['gx','gy','gz']);}

function drawTrainingCurves(h=state.history){if(!h||!h.epoch?.length){drawLineChart($('lossCanvas'),[[],[]],['train','validation']);drawLineChart($('accuracyCanvas'),[[],[]],['train','validation'],{fixedY:[0,1]});return;}drawLineChart($('lossCanvas'),[h.loss,h.valLoss],['train loss','validation loss']);drawLineChart($('accuracyCanvas'),[h.acc,h.valAcc],['train accuracy','validation accuracy'],{fixedY:[0,1]});}

$('clearLogBtn').addEventListener('click',()=>$('log').textContent='');
$('supportNote').textContent=NoodleAIBLE.supportMessage();
refreshLabels();refreshDataset();lockUi(false);updateInputAndTopology();drawLiveEmpty();drawTrainingCurves();initTf();
