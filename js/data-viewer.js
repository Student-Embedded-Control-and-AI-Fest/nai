'use strict';
// NoodleAI read-only NPZ Data Viewer. No changes to capture, model training, or firmware.
// Handles the current NoodleAI NPZ schema and legacy six-axis X layouts.
(() => {
  const $ = id => document.getElementById(id);
  const CHANNELS = [
    ['viewerAx', 0], ['viewerAy', 1], ['viewerAz', 2],
    ['viewerGx', 3], ['viewerGy', 4], ['viewerGz', 5],
  ];
  const ORIENTATION_CHANNELS = [
    ['viewerQw',0,[-1,1]], ['viewerQx',1,[-1,1]],
    ['viewerQy',2,[-1,1]], ['viewerQz',3,[-1,1]],
    ['viewerRoll',4,[-180,180]], ['viewerPitch',5,[-90,90]],
  ];
  let dataset = null;

  // Visual-only spherical interpolation of raw q=[w,x,y,z] samples.
  // Quaternion signs are matched locally because q and -q encode the same pose.
  function unitQuat(data, base) {
    const q=[Number(data[base]),Number(data[base+1]),Number(data[base+2]),Number(data[base+3])];
    const norm=Math.hypot(...q);
    return Number.isFinite(norm)&&norm>1e-10 ? q.map(v=>v/norm) : null;
  }
  function slerp(qa,qb,f) {
    let dot=qa.reduce((s,v,i)=>s+v*qb[i],0);
    if(dot<0){qb=qb.map(v=>-v);dot=-dot;}
    dot=Math.max(-1,Math.min(1,dot));
    if(dot>0.9995){
      const q=qa.map((v,i)=>v+(qb[i]-v)*f), norm=Math.hypot(...q);
      return q.map(v=>v/norm);
    }
    const ang=Math.acos(dot), sinAng=Math.sin(ang);
    const a=Math.sin((1-f)*ang)/sinAng, b=Math.sin(f*ang)/sinAng;
    return qa.map((v,i)=>a*v+b*qb[i]);
  }
  function readOrientation(arrays,n,length) {
    if(!arrays.quat_data) return null;
    const quat=arrays.quat_data, offsets=arrays.raw_offsets;
    if(!offsets || !Array.isArray(offsets.shape) || offsets.shape.length!==1 || offsets.data.length!==n+1)
      throw new Error('quat_data requires raw_offsets with one start index per gesture plus the final endpoint.');
    if(!arrays.raw_data || arrays.raw_data.shape?.length!==2 || arrays.raw_data.shape[1]!==6)
      throw new Error('quat_data requires matching raw_data [total_raw_samples,6].');
    const total=arrays.raw_data.shape[0];
    if(quat.shape?.length!==2 || quat.shape[0]!==total || quat.shape[1]!==4 || quat.data.length!==total*4)
      throw new Error('quat_data must have shape [total_raw_samples,4] with quaternion order [qw,qx,qy,qz].');
    const rows=new Array(n);
    let prevEnd=0;
    for(let i=0;i<n;i++) {
      const start=Number(offsets.data[i]), end=Number(offsets.data[i+1]), srcN=end-start;
      if(!Number.isInteger(start)||!Number.isInteger(end)||start!==prevEnd||srcN<2||end>total)
        throw new Error(`Invalid raw_offsets for gesture ${i+1}.`);
      prevEnd=end;
      const src=new Array(srcN);
      for(let j=0;j<srcN;j++) {
        let q=unitQuat(quat.data,(start+j)*4);
        if(!q) throw new Error(`Invalid quaternion in gesture ${i+1} at sample ${j+1}.`);
        if(j>0 && src[j-1].reduce((sum,v,k)=>sum+v*q[k],0)<0)q=q.map(v=>-v);
        src[j]=q;
      }
      // Canonicalize the initial sign for easier class comparison.
      if(src[0][0]<0) for(let j=0;j<srcN;j++)src[j]=src[j].map(v=>-v);
      const row=new Float32Array(length*6);
      for(let t=0;t<length;t++) {
        const pos=t*(srcN-1)/(length-1), i0=Math.floor(pos), i1=Math.min(i0+1,srcN-1);
        const q=slerp(src[i0],src[i1],pos-i0);
        const [w,x,y,z]=q;
        for(let c=0;c<4;c++)row[t*6+c]=q[c];
        const roll=Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));
        const pitch=Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x))));
        row[t*6+4]=roll*180/Math.PI;
        row[t*6+5]=pitch*180/Math.PI;
      }
      rows[i]=row;
    }
    if(prevEnd!==total)throw new Error('Last raw_offset does not match raw_data length.');
    return rows;
  }

  function firstNumber(arrays, keys, fallback = null) {
    for (const k of keys) {
      const obj = arrays[k];
      if (obj && obj.data && obj.data.length) {
        const v = Number(obj.data[0]);
        if (Number.isFinite(v)) return v;
      }
    }
    return fallback;
  }

  function parseViewerArrays(arrays) {
    const a = arrays.X;
    if (!a || !arrays.y) {
      throw new Error('This dataset needs X.npy and y.npy, with six stored IMU channels.');
    }
    if (!Array.isArray(a.shape) || ![2,3].includes(a.shape.length)) {
      throw new Error('X must be [gestures, time×6], [gestures,time,6], or [gestures,6,time].');
    }
    const n = a.shape[0];
    let length, format;
    if (a.shape.length === 2) {
      const d = a.shape[1];
      if (d % 6) throw new Error(`X width ${d} is not divisible by six; cannot display six channels.`);
      length = d / 6;
      format = 'flat';
    } else if (a.shape[2] === 6) {
      length = a.shape[1];
      format = 'time-major';
    } else if (a.shape[1] === 6) {
      length = a.shape[2];
      format = 'channel-major';
    } else {
      throw new Error(`Expected six axes in X; got shape (${a.shape.join(', ')}).`);
    }
    if (!Number.isInteger(n) || n < 1 || length < 2 || n > 250000) throw new Error('Invalid gesture count or window length.');
    const storedLength = firstNumber(arrays, ['normalized_length', 'window_length']);
    if (storedLength !== null && storedLength !== length)
      throw new Error(`Stored window length ${storedLength} does not match X shape (${length}).`);
    const yObj = arrays.y;
    const rawY = Array.from(yObj.data, v => String(v).trim());
    if (rawY.length !== n) throw new Error(`X contains ${n} gestures but y contains ${rawY.length} labels.`);

    const labelArray = arrays.labels ? Array.from(arrays.labels.data, v => String(v)) : [];
    const unique = [...new Set(rawY)];
    let orderedLabels, mapped;
    const allInts = rawY.every(v => /^-?\d+$/.test(v));
    if (allInts) {
      const nums = rawY.map(Number);
      const uniqNums = [...new Set(nums)].sort((a,b) => a-b);
      const isDirect = labelArray.length > 0 && nums.every(v => v >= 0 && v < labelArray.length);
      const isOneBased = labelArray.length > 0 && nums.every(v => v >= 1 && v <= labelArray.length) && !nums.includes(0);
      if (isDirect) {
        orderedLabels = labelArray;
        mapped = nums;
      } else if (isOneBased) {
        orderedLabels = labelArray;
        mapped = nums.map(v => v-1);
      } else {
        orderedLabels = uniqNums.map((v,i) => labelArray.length === uniqNums.length ? labelArray[i] : `Class ${v}`);
        mapped = nums.map(v => uniqNums.indexOf(v));
      }
    } else {
      orderedLabels = labelArray.length ? labelArray : unique;
      for (const v of unique) if (!orderedLabels.includes(v)) orderedLabels.push(v);
      mapped = rawY.map(v => orderedLabels.indexOf(v));
    }
    if (orderedLabels.length === 0) throw new Error('No gesture classes found in y.');
    const width = length * 6;
    if (a.data.length !== n*width) throw new Error('X data length does not match declared shape.');

    const rows = new Array(n);
    const rangeMin = [Infinity, Infinity];
    const rangeMax = [-Infinity, -Infinity];
    for (let i=0; i<n; i++) {
      const row = new Float32Array(width);
      for (let t=0; t<length; t++) {
        for (let c=0; c<6; c++) {
          const flatIndex = i*width + (format === 'channel-major' ? c*length+t : t*6+c);
          const v = Number(a.data[flatIndex]);
          row[t*6+c] = Number.isFinite(v) ? v : NaN;
          if (Number.isFinite(v)) {
            const group = c < 3 ? 0 : 1;
            if (v < rangeMin[group]) rangeMin[group] = v;
            if (v > rangeMax[group]) rangeMax[group] = v;
          }
        }
      }
      rows[i] = row;
    }
    const axis = [0,1].map(g => {
      let lo = rangeMin[g], hi = rangeMax[g];
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-1,1];
      // Include zero so amplitude and polarity are comparable across classes.
      lo = Math.min(0, lo); hi = Math.max(0, hi);
      const spread = hi-lo;
      const pad = spread === 0 ? 1 : spread * 0.055;
      return [lo-pad, hi+pad];
    });
    const classRows = orderedLabels.map(() => []);
    for (let i=0; i<n; i++) classRows[mapped[i]].push(i);
    let orientationRows=null, orientationWarning=null;
    if(arrays.quat_data) {
      try { orientationRows=readOrientation(arrays,n,length); }
      catch(error) { orientationWarning=error.message; } // Preserve six-channel inspection.
    }
    return { labels: orderedLabels, classRows, rows, orientationRows, orientationWarning, length, axis,
      rateHz: firstNumber(arrays, ['sample_rate_hz'], null), count: n };
  }

  function status(message, isError = false) {
    const el = $('viewerStatus');
    el.textContent = message;
    el.classList.toggle('error', isError);
  }
  function setText(id, content) { $(id).textContent = content; }

  function redrawPanel(canvas, d, c, seriesIndices, mean, axis, onlyOne, seriesRows=d.rows, stroke='#3086d1', noMean=false) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = {l:59, r:13, t:14, b:33};
    const plotW = W-pad.l-pad.r, plotH = H-pad.t-pad.b;
    const [lo,hi] = axis;
    const xx = t => pad.l + t / Math.max(1,d.length-1) * plotW;
    const yy = v => pad.t + (hi-v)/(hi-lo) * plotH;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = '#fbfcfe'; ctx.fillRect(0,0,W,H);
    ctx.save();ctx.strokeStyle='#e5ebf2';ctx.lineWidth=1;
    for(let k=0;k<=4;k++) {
      const y=pad.t + k*plotH/4;
      ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    }
    for(let k=0;k<=4;k++) {
      const x=pad.l + k*plotW/4;
      ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();
    }
    if(lo<=0 && hi>=0) {
      ctx.strokeStyle='#bbcbdf';ctx.setLineDash([5,4]);ctx.beginPath();
      ctx.moveTo(pad.l,yy(0));ctx.lineTo(W-pad.r,yy(0));ctx.stroke();ctx.setLineDash([]);
    }
    ctx.restore();
    ctx.font='12px system-ui, sans-serif';ctx.textAlign='right';ctx.fillStyle='#76879a';
    for(let k=0;k<=4;k++) {
      const v=hi-k*(hi-lo)/4;
      ctx.fillText((Math.abs(v)<0.005?0:v).toFixed(Math.abs(hi-lo)>=100?0:2),pad.l-8,pad.t+k*plotH/4+4);
    }
    ctx.textAlign='center';
    for(let k=0;k<=4;k++) {
      const step=Math.round(k*(d.length-1)/4);
      ctx.fillText(String(step),pad.l+k*plotW/4,H-17);
    }
    ctx.font='10px system-ui';ctx.fillText('sample index',pad.l+plotW/2,H-3);
    ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,plotW,plotH);ctx.clip();
    const draw = (row, color, width, alpha, meanVector = false) => {
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.globalAlpha=alpha;
      ctx.beginPath();let active=false;
      for (let t=0;t<d.length;t++) {
        const v = meanVector ? row[t] : row[t*6+c];
        if(!Number.isFinite(v)) {active=false;continue;}
        if(!active) {ctx.moveTo(xx(t),yy(v));active=true;} else ctx.lineTo(xx(t),yy(v));
      }
      ctx.stroke();
    };
    for (const row of seriesIndices) {
      draw(seriesRows[row], noMean?stroke:(c<3?'#77a8eb':'#a79ada'), onlyOne?2:1, onlyOne?0.95:(noMean?0.45:0.25));
    }
    if(!onlyOne && !noMean) draw(mean,c<3?'#1759c0':'#6d4fb8',2.7,1,true);
    ctx.restore();
  }

  function redraw() {
    if (!dataset) return;
    const cls = Number($('viewerClass').value);
    const ids = dataset.classRows[cls] || [];
    const sampleChoice = $('viewerSample').value;
    const one = sampleChoice !== 'all';
    const plotIndices = one ? [Number(sampleChoice)] : (ids.length <= 32 ? ids : Array.from({length:32},(_,k)=>ids[Math.floor(k*(ids.length-1)/31)]));
    const mean = Array.from({length:6},()=>new Float64Array(dataset.length));
    if (!one) {
      const counts = Array.from({length:6},()=>new Uint32Array(dataset.length));
      for (const id of ids) {
        const r=dataset.rows[id];
        for(let t=0;t<dataset.length;t++) for(let c=0;c<6;c++) {
          const v = r[t*6+c];
          if (Number.isFinite(v)) {mean[c][t]+=v;counts[c][t]++;}
        }
      }
      for (let c=0;c<6;c++) for (let t=0;t<dataset.length;t++) {
        mean[c][t] = counts[c][t] ? mean[c][t]/counts[c][t] : NaN;
      }
    }
    for(const [id,c] of CHANNELS){
      const canvas=$(id);
      redrawPanel(canvas,dataset,c,plotIndices,mean[c],dataset.axis[c<3?0:1],one);
    }
    if(dataset.orientationRows){
      for(const [id,c,limits] of ORIENTATION_CHANNELS)
        redrawPanel($(id),dataset,c,plotIndices,null,limits,one,dataset.orientationRows,c<4?'#128477':'#9057b0',true);
    }
    setText('viewerSelectedCount', String(ids.length));
    const chosenName = dataset.labels[cls] || `Class ${cls}`;
    const oriInfo=dataset.orientationRows
      ? ' Optional orientation plots available below.'
      : dataset.orientationWarning
        ? ` Orientation was skipped: ${dataset.orientationWarning}` : '';
    status(one
      ? `${chosenName}: viewing gesture ${ids.indexOf(Number(sampleChoice))+1} of ${ids.length}. Y scales remain locked across classes.${oriInfo}`
      : `${chosenName}: ${ids.length} gestures. Bold line is the mean of every IMU channel; ${plotIndices.length} individual traces shown. Y scales are locked across classes.${oriInfo}`);
  }

  function populate() {
    const sel = $('viewerClass');sel.innerHTML='';
    for(let i=0;i<dataset.labels.length;i++) {
      const opt=document.createElement('option');opt.value=String(i);
      opt.textContent=`${dataset.labels[i]} (${dataset.classRows[i].length})`;
      sel.appendChild(opt);
    }
    const firstNonempty = dataset.classRows.findIndex(rows => rows.length);
    sel.selectedIndex=firstNonempty>=0?firstNonempty:0;
    sel.disabled=false;
    $('viewerStats').hidden=false;
    $('viewerOrientationSection').hidden=!dataset.orientationRows;
    $('viewerOrientationSection').open=false; // Stay collapsed when a new dataset loads.
    setText('viewerOrientationStatus',dataset.orientationRows
      ? 'Continuous device quaternion found · 4 components + derived roll/pitch'
      : '');
    setText('viewerTotal',String(dataset.count));
    setText('viewerClasses',String(dataset.labels.length));
    setText('viewerSteps',String(dataset.length));
    makeSampleOptions();
    redraw();
  }

  function makeSampleOptions() {
    const select=$('viewerSample'), cls=Number($('viewerClass').value);
    const ids=dataset.classRows[cls] || [];
    select.innerHTML='';
    const all=document.createElement('option');
    all.value='all';all.textContent='All gestures · mean + examples';select.appendChild(all);
    ids.forEach((row,i)=>{
      const opt=document.createElement('option');opt.value=String(row);
      opt.textContent=`Gesture ${i+1} (dataset row ${row+1})`;
      select.appendChild(opt);
    });
    select.disabled=!ids.length;
    select.value='all';
  }

  async function openDataset(file) {
    if(!file) return;
    status(`Opening ${file.name} locally…`);
    try {
      if(!/\.npz$/i.test(file.name)) throw new Error('Choose a .npz dataset.');
      if(file.size > 200*1024*1024) throw new Error('Dataset exceeds the 200 MiB viewer limit.');
      if(!window.NAI?.loadNpz) throw new Error('NoodleAI dataset reader has not loaded.');
      const arrays=await NAI.loadNpz(file);
      const loaded=parseViewerArrays(arrays);
      dataset=loaded;
      setText('viewerFileName', file.name);
      populate();
    } catch(e) {
      status(`Cannot open “${file.name}”: ${e.message}`,true);
    }
  }

  $('viewerLoadBtn').addEventListener('click',()=>$('viewerFileInput').click());
  $('viewerFileInput').addEventListener('change',event=>{
    const file=event.target.files && event.target.files[0];
    if(file)void openDataset(file);
    event.target.value='';
  });
  $('viewerClass').addEventListener('change',()=>{if(dataset){makeSampleOptions();redraw();}});
  $('viewerSample').addEventListener('change',redraw);

  // Visibility changes do not recalculate domains, and chart sizes use fixed
  // intrinsic coordinates so locked limits are never lost on window resize.
  window.NAIDataViewer = Object.freeze({ parseViewerArrays, readOrientation, openDataset });
})();
