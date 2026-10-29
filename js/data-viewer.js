'use strict';
// NoodleAI read-only NPZ Data Viewer. No changes to capture, model training, or firmware.
// Handles the current NoodleAI NPZ schema and legacy six-axis X layouts.
(() => {
  const $ = id => document.getElementById(id);
  const CHANNELS = [
    ['viewerAx', 0], ['viewerAy', 1], ['viewerAz', 2],
    ['viewerGx', 3], ['viewerGy', 4], ['viewerGz', 5],
  ];
  let dataset = null;

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
    return { labels: orderedLabels, classRows, rows, length, axis,
      rateHz: firstNumber(arrays, ['sample_rate_hz'], null), count: n };
  }

  function status(message, isError = false) {
    const el = $('viewerStatus');
    el.textContent = message;
    el.classList.toggle('error', isError);
  }
  function setText(id, content) { $(id).textContent = content; }

  function redrawPanel(canvas, d, c, seriesIndices, mean, axis, onlyOne) {
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
      draw(d.rows[row], c<3?'#77a8eb':'#a79ada', onlyOne?2:1, onlyOne?0.95:0.25);
    }
    if(!onlyOne) draw(mean,c<3?'#1759c0':'#6d4fb8',2.7,1,true);
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
    setText('viewerSelectedCount', String(ids.length));
    const chosenName = dataset.labels[cls] || `Class ${cls}`;
    status(one
      ? `${chosenName}: viewing gesture ${ids.indexOf(Number(sampleChoice))+1} of ${ids.length}. Y scales remain locked across classes.`
      : `${chosenName}: ${ids.length} gestures. Bold line is the mean of every gesture; ${plotIndices.length} individual traces shown. Y scales are locked across classes.`);
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
  window.NAIDataViewer = Object.freeze({ parseViewerArrays, openDataset });
})();
