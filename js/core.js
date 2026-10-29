'use strict';

const SAMPLE_RATE_HZ = 50;
const MADGWICK_BETA = 0.10;
const GRAVITY_MPS2 = 9.80665;
const MAX_PACKAGE_BYTES = 550 * 1024;
const FLAG_BINARY_CLASSIFIER = 0x0001;
const FLAG_INT8 = 0x0002;
const FLAG_LPF5 = 0x0008; // Required for all new NAI4 models: 5 Hz LPF before resampling.
const FLAG_CONT_ORIENTATION = 0x0010; // Quaternion estimated continuously on device, never reset by BOOT.
const QCFG_MAGIC = 'Q8M1';

const REP = {
  'accel': { code: 1, channels: 3 },
  'gyro': { code: 2, channels: 3 },
  'accel+gyro': { code: 3, channels: 6 },
  'quaternion': { code: 4, channels: 4 },
  'velocity': { code: 5, channels: 3 },
  'velocity+quaternion': { code: 6, channels: 7 },
  'accel-dir+mag': { code: 9, channels: 4 },
  'accel+roll+pitch': { code: 10, channels: 5 }, // unit direction + LPF acceleration magnitude
};
const CODE_TO_REP = Object.fromEntries(Object.entries(REP).map(([k,v]) => [v.code, k]));
// Acceleration in g. Directions below this magnitude are set to zero, not amplified.
const ACCEL_DIRECTION_EPS_G = 0.001;
// ---------------------------------------------------------------------------
// NPY / NPZ reader: enough for NoodleAI .npz datasets
// ---------------------------------------------------------------------------

function product(shape) {
  return shape.length ? shape.reduce((a,b) => a*b, 1) : 1;
}

function parseShape(header) {
  const m = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
  if (!m) throw new Error('NPY shape not found');
  const text = m[1].trim();
  if (!text) return [];
  return text.split(',').map(s => s.trim()).filter(Boolean).map(Number);
}

function parseNpy(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  if (u8.length < 10 || u8[0] !== 0x93 ||
      String.fromCharCode(...u8.slice(1,6)) !== 'NUMPY') {
    throw new Error('Not an NPY file');
  }
  const major = u8[6], minor = u8[7];
  const dv = new DataView(arrayBuffer);
  let headerLen, pos;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    pos = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    pos = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.${minor}`);
  }
  const header = new TextDecoder('latin1').decode(u8.slice(pos, pos + headerLen));
  const descrM = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  const fortM = header.match(/['"]fortran_order['"]\s*:\s*(True|False)/);
  if (!descrM || !fortM) throw new Error('Malformed NPY header');
  if (fortM[1] === 'True') throw new Error('Fortran-order NPY arrays are not supported');
  const descr = descrM[1];
  const shape = parseShape(header);
  const count = product(shape);
  const dataOffset = pos + headerLen;
  const d = new DataView(arrayBuffer, dataOffset);

  let data;
  const endian = descr[0];
  const body = ['<','>','|','='].includes(endian) ? descr.slice(1) : descr;
  const little = endian !== '>';

  if (body === 'f4') {
    data = new Float32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getFloat32(i*4, little);
  } else if (body === 'f8') {
    data = new Float64Array(count);
    for (let i=0;i<count;i++) data[i] = d.getFloat64(i*8, little);
  } else if (body === 'i4') {
    data = new Int32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getInt32(i*4, little);
  } else if (body === 'u4') {
    data = new Uint32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getUint32(i*4, little);
  } else if (body === 'i8') {
    data = new Array(count);
    for (let i=0;i<count;i++) data[i] = Number(d.getBigInt64(i*8, little));
  } else if (body === 'u8') {
    data = new Array(count);
    for (let i=0;i<count;i++) data[i] = Number(d.getBigUint64(i*8, little));
  } else if (body === 'i2') {
    data = new Int16Array(count);
    for (let i=0;i<count;i++) data[i] = d.getInt16(i*2, little);
  } else if (body === 'u2') {
    data = new Uint16Array(count);
    for (let i=0;i<count;i++) data[i] = d.getUint16(i*2, little);
  } else if (body === 'i1') {
    data = new Int8Array(arrayBuffer, dataOffset, count).slice();
  } else if (body === 'u1' || body === 'b1') {
    data = new Uint8Array(arrayBuffer, dataOffset, count).slice();
  } else if (body.startsWith('U')) {
    const chars = Number(body.slice(1));
    data = new Array(count);
    let off = 0;
    for (let i=0;i<count;i++) {
      let s = '';
      for (let j=0;j<chars;j++) {
        const cp = d.getUint32(off, little);
        off += 4;
        if (cp) s += String.fromCodePoint(cp);
      }
      data[i] = s;
    }
  } else if (body.startsWith('S')) {
    const width = Number(body.slice(1));
    data = new Array(count);
    let off = 0;
    for (let i=0;i<count;i++) {
      const bytes = new Uint8Array(arrayBuffer, dataOffset + off, width);
      off += width;
      let n = bytes.indexOf(0);
      if (n < 0) n = width;
      data[i] = new TextDecoder('utf-8').decode(bytes.slice(0,n));
    }
  } else {
    throw new Error(`Unsupported NPY dtype ${descr}`);
  }

  return { data, shape, descr };
}

async function loadNpz(file) {
  const zip = await JSZip.loadAsync(await file.arrayBuffer());
  const arrays = {};
  for (const [name, entry] of Object.entries(zip.files)) {
    if (entry.dir || !name.endsWith('.npy')) continue;
    const key = name.replace(/\.npy$/, '');
    arrays[key] = parseNpy(await entry.async('arraybuffer'));
  }
  return arrays;
}

function scalar(arrays, key, fallback=null) {
  if (!arrays[key]) return fallback;
  const d = arrays[key].data;
  return Array.isArray(d) || ArrayBuffer.isView(d) ? d[0] : d;
}

function rowFromFlat(flat, row, width) {
  return Float32Array.from(flat.subarray(row*width, (row+1)*width));
}

function parseDatasetArrays(a) {
  if (!a.X || !a.y || !a.labels) throw new Error('Dataset must contain X, y, and labels');
  const Xshape = a.X.shape;
  if (Xshape.length !== 2) throw new Error(`X must be 2-D; got shape ${Xshape}`);
  const n = Xshape[0], width = Xshape[1];

  const y = Int32Array.from(a.y.data, Number);
  if (y.length !== n) throw new Error('X/y row count mismatch');

  const labels = Array.from(a.labels.data, String);
  const N = Number(scalar(a, 'normalized_length', scalar(a, 'window_length', 0)));
  if (!N) throw new Error('Dataset lacks normalized_length/window_length');
  if (width !== N*6) throw new Error(`Expected stored six-axis X width ${N*6}; got ${width}`);

  const Xrows = Array.from({length:n}, (_,i) => rowFromFlat(a.X.data, i, width));

  let rawRows = null;
  if (a.raw_data && a.raw_offsets) {
    const rs = a.raw_data.shape;
    if (rs.length !== 2 || rs[1] !== 6) throw new Error('raw_data must have shape [T,6]');
    const offsets = Array.from(a.raw_offsets.data, Number);
    if (offsets.length !== n+1) throw new Error('raw_offsets must have n_samples+1 entries');
    rawRows = [];
    for (let i=0;i<n;i++) {
      const start = offsets[i], end = offsets[i+1];
      if (end-start < 2) throw new Error(`Raw gesture ${i} has fewer than 2 samples`);
      const row = new Float32Array((end-start)*6);
      for (let k=start;k<end;k++) {
        for (let c=0;c<6;c++) row[(k-start)*6+c] = a.raw_data.data[k*6+c];
      }
      rawRows.push({ data: row, length: end-start });
    }
  }

  if (a.quat_data) {
    if (!rawRows) throw new Error('quat_data requires raw_data and raw_offsets');
    if (a.quat_data.shape.length!==2 || a.quat_data.shape[1]!==4 ||
        a.quat_data.shape[0]!==a.raw_data.shape[0])
      throw new Error('quat_data must be [total_raw_samples,4]');
    let row=0;
    for (const r of rawRows) {
      r.quat=Float32Array.from(a.quat_data.data.slice(row*4,(row+r.length)*4));
      row+=r.length;
    }
  }
  const sampleRate = Number(scalar(a, 'sample_rate_hz', SAMPLE_RATE_HZ));
  return { Xrows, y, labels, N, sampleRate, rawRows, width, fileName: '' };
}


// ---------------------------------------------------------------------------
// Exact legacy NumPy RandomState MT19937 permutation and sklearn-style split
// ---------------------------------------------------------------------------

class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.index = 624;
    this.mt[0] = seed >>> 0;
    for (let i=1;i<624;i++) {
      const x = this.mt[i-1] ^ (this.mt[i-1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, x) + i) >>> 0;
    }
  }
  twist() {
    for (let i=0;i<624;i++) {
      const y = (this.mt[i] & 0x80000000) | (this.mt[(i+1)%624] & 0x7fffffff);
      this.mt[i] = this.mt[(i+397)%624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
    }
    this.index = 0;
  }
  uint32() {
    if (this.index >= 624) this.twist();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  interval(max) {
    max >>>= 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4;
    mask |= mask >>> 8; mask |= mask >>> 16;
    let v;
    do { v = this.uint32() & mask; } while (v > max);
    return v >>> 0;
  }
  permutationArray(values) {
    const a = Array.from(values);
    for (let i=a.length-1;i>0;i--) {
      const j = this.interval(i);
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
}

function approximateMode(classCounts, nDraws, rng) {
  const total = classCounts.reduce((a,b)=>a+b,0);
  const continuous = classCounts.map(c => c/total*nDraws);
  const floored = continuous.map(Math.floor);
  let need = nDraws - floored.reduce((a,b)=>a+b,0);
  if (need > 0) {
    const remainder = continuous.map((v,i)=>v-floored[i]);
    const values = Array.from(new Set(remainder)).sort((a,b)=>b-a);
    for (const value of values) {
      const inds = [];
      for (let i=0;i<remainder.length;i++) {
        if (Math.abs(remainder[i]-value) < 1e-15) inds.push(i);
      }
      const addNow = Math.min(inds.length, need);
      const chosen = addNow === inds.length ? inds : rng.permutationArray(inds).slice(0, addNow);
      for (const i of chosen) floored[i] += 1;
      need -= addNow;
      if (need === 0) break;
    }
  }
  return floored;
}

function pythonRound(x) {
  const f = Math.floor(x), frac = x-f;
  if (frac < 0.5) return f;
  if (frac > 0.5) return f+1;
  return (f % 2 === 0) ? f : f+1;
}

function sklearnStratifiedSplit(y, nClasses, seed=42) {
  const n = y.length;
  const nTest = Math.max(nClasses, pythonRound(0.20*n));
  const nTrain = n - nTest;
  const byClass = Array.from({length:nClasses}, ()=>[]);
  for (let i=0;i<n;i++) byClass[y[i]].push(i);
  const counts = byClass.map(v=>v.length);
  if (Math.min(...counts) < 2) throw new Error('Every class needs at least 2 samples for stratification');

  const rng = new MT19937(seed);
  const n_i = approximateMode(counts, nTrain, rng);
  const remain = counts.map((c,i)=>c-n_i[i]);
  const t_i = approximateMode(remain, nTest, rng);

  let train = [], test = [];
  for (let c=0;c<nClasses;c++) {
    const perm = rng.permutationArray(Array.from({length:counts[c]}, (_,i)=>i));
    const idxs = perm.map(p=>byClass[c][p]);
    train.push(...idxs.slice(0, n_i[c]));
    test.push(...idxs.slice(n_i[c], n_i[c]+t_i[c]));
  }
  train = rng.permutationArray(train);
  test = rng.permutationArray(test);
  return { train, test };
}


// ---------------------------------------------------------------------------
// NAI4 motion representations — mirror the current Python/C++ implementation
// ---------------------------------------------------------------------------

function resample(data, rows, cols, targetN) {
  const out = new Float32Array(targetN*cols);
  const scale = (rows-1)/(targetN-1);
  for (let i=0;i<targetN;i++) {
    const pos = i*scale;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0+1, rows-1);
    const a = pos-i0;
    for (let c=0;c<cols;c++) {
      out[i*cols+c] = data[i0*cols+c] + a*(data[i1*cols+c]-data[i0*cols+c]);
    }
  }
  return out;
}

function centerColumns(data, rows, cols) {
  const out = Float32Array.from(data);
  for (let c=0;c<cols;c++) {
    let s=0;
    for (let r=0;r<rows;r++) s += out[r*cols+c];
    const m=s/rows;
    for (let r=0;r<rows;r++) out[r*cols+c] -= m;
  }
  return out;
}

function qNorm(q) {
  const n = Math.hypot(q[0],q[1],q[2],q[3]);
  if (!Number.isFinite(n) || n < 1e-12) return [1,0,0,0];
  return q.map(v=>v/n);
}
function qConj(q) { return [q[0],-q[1],-q[2],-q[3]]; }
function qMul(a,b) {
  const [aw,ax,ay,az]=a, [bw,bx,by,bz]=b;
  return [
    aw*bw-ax*bx-ay*by-az*bz,
    aw*bx+ax*bw+ay*bz-az*by,
    aw*by-ax*bz+ay*bw+az*bx,
    aw*bz+ax*by-ay*bx+az*bw,
  ];
}
function qRotate(q,v) {
  return qMul(qMul(q,[0,v[0],v[1],v[2]]),qConj(q)).slice(1);
}
function dot(a,b) { return a.reduce((s,v,i)=>s+v*b[i],0); }

// Derived gestures use the same quaternion samples computed by the device
// during acquisition, including motion between BOOT holds. No local Madgwick
// reinitialization is permitted for continuous-orientation representations.
function deriveContinuousFeatures(filtered, quat4, T, rep, sampleRate) {
  if (!quat4 || quat4.length!==T*4)
    throw new Error('This input kind requires saved continuous Madgwick quaternions. Record new gestures with updated firmware.');
  const ch=REP[rep].channels, out=new Float32Array(T*ch), dt=1/sampleRate;
  let prevQ=null, prevA=null, velocity=[0,0,0];
  for(let k=0;k<T;k++){
    const r=k*6,d=k*ch;
    let q=qNorm(Array.from(quat4.slice(k*4,k*4+4)));
    if(prevQ && dot(q,prevQ)<0)q=q.map(x=>-x);
    prevQ=q;
    if(rep==='accel+roll+pitch'){
      const [w,x,y,z]=q;
      out[d]=filtered[r];out[d+1]=filtered[r+1];out[d+2]=filtered[r+2];
      out[d+3]=Math.fround(Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)));
      out[d+4]=Math.fround(Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x)))));
      continue;
    }
    if(rep==='quaternion'||rep==='velocity+quaternion') {
      out.set(q,d+(rep==='quaternion'?0:3));
    }
    if(rep==='velocity'||rep==='velocity+quaternion'){
      const aw=qRotate(q,[filtered[r],filtered[r+1],filtered[r+2]]);
      const curr=[aw[0]*GRAVITY_MPS2,aw[1]*GRAVITY_MPS2,(aw[2]-1)*GRAVITY_MPS2];
      if(prevA)for(let c=0;c<3;c++)velocity[c]+=0.5*(prevA[c]+curr[c])*dt;
      prevA=curr;
      for(let c=0;c<3;c++)out[d+c]=Math.fround(velocity[c]);
    }
  }
  if(rep==='velocity'||rep==='velocity+quaternion'){
    const end=Array.from(out.slice((T-1)*ch,(T-1)*ch+3));
    for(let k=0;k<T;k++)for(let c=0;c<3;c++)out[k*ch+c]=Math.fround(out[k*ch+c]-(k/(T-1))*end[c]);
    for(let c=0;c<3;c++){out[c]=0;out[(T-1)*ch+c]=0;}
  }
  return out;
}

function normalizeQuatRows(data, rows, offset=0, stride=4) {
  const out=Float32Array.from(data);
  for (let r=0;r<rows;r++) {
    const b=r*stride+offset;
    const q=qNorm([out[b],out[b+1],out[b+2],out[b+3]]);
    out[b]=q[0];out[b+1]=q[1];out[b+2]=q[2];out[b+3]=q[3];
  }
  return out;
}

// First-order LPF, 5 Hz, applied independently to each requested raw IMU axis.
// Per-gesture reset: LP[0] = raw[0], so a stationary accelerometer preserves 1 g.
function filterRaw6Lpf5(raw, T, sampleRate, axes=[0,1,2,3,4,5]) {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error('Invalid sensor sample rate');
  const out=Float32Array.from(raw);
  const dt=1/sampleRate, tau=1/(2*Math.PI*5.0);
  const beta=Math.fround(dt/(tau+dt));
  for (const c of axes) {
    let lp=out[c];
    for (let t=1;t<T;t++) {
      const o=t*6+c;
      const x=out[o];
      lp=Math.fround(lp+Math.fround(beta*Math.fround(x-lp)));
      out[o]=lp;
    }
  }
  return out;
}

function buildRepresentationOne(ds, sampleIndex, rep) {
  const N=ds.N;
  if (!REP[rep]) throw new Error(`Unknown representation ${rep}`);
  // LPF5 requires raw, non-resampled IMU data. Resampling happens afterward.
  if (!ds.rawRows) throw new Error('LPF5 requires original raw six-axis samples (raw_data/raw_offsets). Record new gestures or load a raw NAI4 dataset.');
  const rawObj=ds.rawRows[sampleIndex];
  const T=rawObj.length;
  const idx=rep==='gyro'?[3,4,5]:rep==='accel+gyro'?[0,1,2,3,4,5]:[0,1,2];
  const filtered=filterRaw6Lpf5(rawObj.data,T,ds.sampleRate,rep==='quaternion'||rep==='velocity'||rep==='velocity+quaternion'||rep==='accel+roll+pitch'?[0,1,2,3,4,5]:idx);
  if (rep==='accel'||rep==='gyro'||rep==='accel+gyro'||rep==='accel-dir+mag') {
    const selected=new Float32Array(T*idx.length);
    for(let r=0;r<T;r++)for(let c=0;c<idx.length;c++)selected[r*idx.length+c]=filtered[r*6+idx[c]];
    const xyz=resample(selected,T,idx.length,N);
    if(rep==='accel-dir+mag') {
      const out=new Float32Array(N*4);
      for(let r=0;r<N;r++) {
        const i=r*3,o=r*4,ax=xyz[i],ay=xyz[i+1],az=xyz[i+2];
        const m=Math.fround(Math.sqrt(ax*ax+ay*ay+az*az));
        const inv=(Number.isFinite(m)&&m>ACCEL_DIRECTION_EPS_G)?1/m:0;
        out[o]=Math.fround(ax*inv);out[o+1]=Math.fround(ay*inv);out[o+2]=Math.fround(az*inv);
        out[o+3]=Number.isFinite(m)?m:0;
      }
      return out; // Do not center: preserve direction and approximately 1 g at rest.
    }
    return centerColumns(xyz,N,idx.length); // Center each direct IMU axis per gesture.
  }
  if (!rawObj.quat || rawObj.quat.length!==T*4)
    throw new Error(`${rep} needs continuous quaternion samples. Existing raw-only datasets cannot reproduce the device state; record new gestures.`);
  const continuous=deriveContinuousFeatures(filtered,rawObj.quat,T,rep,ds.sampleRate);
  const channels=REP[rep].channels;
  const converted=resample(continuous,T,channels,N);
  if(rep==='quaternion')return normalizeQuatRows(converted,N);
  if(rep==='velocity+quaternion')return normalizeQuatRows(converted,N,3,7);
  return converted;

}

function buildRepresentationDataset(ds, rep) {
  return Array.from({length:ds.y.length},(_,i)=>buildRepresentationOne(ds,i,rep));
}


// ---------------------------------------------------------------------------
// StandardScaler exactly in the spirit of sklearn: train only, ddof=0
// ---------------------------------------------------------------------------

function fitScaler(rows, trainIdx) {
  const D=rows[0].length, n=trainIdx.length;
  const mean=new Float64Array(D), variance=new Float64Array(D);
  for (const i of trainIdx) {
    const x=rows[i];
    for (let j=0;j<D;j++) mean[j]+=x[j];
  }
  for (let j=0;j<D;j++) mean[j]/=n;
  for (const i of trainIdx) {
    const x=rows[i];
    for (let j=0;j<D;j++) {
      const d=x[j]-mean[j];
      variance[j]+=d*d;
    }
  }
  const scale=new Float64Array(D);
  for (let j=0;j<D;j++) {
    variance[j]/=n;
    const s=Math.sqrt(variance[j]);
    scale[j]=(Number.isFinite(s)&&s>1e-12)?s:1.0;
  }
  return {mean,scale};
}

// Preserve unit XYZ direction at the actual FCN input for NAI4 rep 9.
// Only magnitude channel is standardized. Direction remains a unit vector.
function configureRepresentationScaler(scaler, rep, N) {
  if (rep==='accel-dir+mag') {
    if (scaler.mean.length!==N*4 || scaler.scale.length!==N*4) {
      throw new Error('Direction + magnitude scaler must have 4*N features');
    }
    for (let t=0;t<N;t++) for (let c=0;c<3;c++) {
      scaler.mean[4*t+c]=0;
      scaler.scale[4*t+c]=1;
    }
  }
  return scaler;
}

function standardizeRows(rows, indices, scaler) {
  const D=rows[0].length;
  return indices.map(i=>{
    const out=new Float32Array(D), x=rows[i];
    for (let j=0;j<D;j++) out[j]=(x[j]-scaler.mean[j])/scaler.scale[j];
    return out;
  });
}

function flattenRows(rows) {
  const n=rows.length,D=rows[0].length,out=new Float32Array(n*D);
  rows.forEach((r,i)=>out.set(r,i*D));
  return out;
}



function makeModel(inputDim, hidden, nClasses, nTrain) {
  const model=tf.sequential();
  const alpha=0.0001; // sklearn MLPClassifier default
  const l2=alpha/(2*nTrain); // match sklearn's alpha/(2*n_samples) loss term

  hidden.forEach((units,i)=>{
    model.add(tf.layers.dense({
      inputShape: i===0 ? [inputDim] : undefined,
      units,
      activation:'relu',
      kernelInitializer: tf.initializers.glorotUniform({seed:42+i}),
      biasInitializer:'zeros',
      kernelRegularizer: tf.regularizers.l2({l2}),
    }));
  });

  if (nClasses===2) {
    model.add(tf.layers.dense({
      units:1, activation:'sigmoid',
      kernelInitializer:tf.initializers.glorotUniform({seed:42+hidden.length}),
      biasInitializer:'zeros',
      kernelRegularizer:tf.regularizers.l2({l2}),
    }));
    model.compile({
      optimizer:tf.train.adam(0.001,0.9,0.999,1e-8),
      loss:'binaryCrossentropy',
      metrics:['accuracy'],
    });
  } else {
    model.add(tf.layers.dense({
      units:nClasses, activation:'softmax',
      kernelInitializer:tf.initializers.glorotUniform({seed:42+hidden.length}),
      biasInitializer:'zeros',
      kernelRegularizer:tf.regularizers.l2({l2}),
    }));
    model.compile({
      optimizer:tf.train.adam(0.001,0.9,0.999,1e-8),
      loss:'categoricalCrossentropy',
      metrics:['accuracy'],
    });
  }
  return model;
}

function argmaxRow(prob) {
  let bi=0,bv=prob[0];
  for (let i=1;i<prob.length;i++) if (prob[i]>bv){bv=prob[i];bi=i;}
  return bi;
}
function accuracy(pred,y){let n=0;for(let i=0;i<y.length;i++)if(pred[i]===y[i])n++;return n/y.length;}

async function predictTf(model, standardizedRows, nClasses) {
  const n=standardizedRows.length,D=standardizedRows[0].length;
  const xs=tf.tensor2d(flattenRows(standardizedRows),[n,D],'float32');
  const out=model.predict(xs);
  const raw=await out.data();
  xs.dispose(); out.dispose();
  const probs=[], pred=[];
  if (nClasses===2) {
    for(let i=0;i<n;i++){const p=raw[i];probs.push([1-p,p]);pred.push(p>=0.5?1:0);}
  } else {
    for(let i=0;i<n;i++){const p=Array.from(raw.slice(i*nClasses,(i+1)*nClasses));probs.push(p);pred.push(argmaxRow(p));}
  }
  return {probs,pred};
}


// ---------------------------------------------------------------------------
// Binary NAI4 writer / reader
// ---------------------------------------------------------------------------

function float32LE(values) {
  const out=new ArrayBuffer(values.length*4),dv=new DataView(out);
  for(let i=0;i<values.length;i++)dv.setFloat32(i*4,Number(values[i]),true);
  return new Uint8Array(out);
}
function readFloat32LE(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), out=new Float32Array(bytes.byteLength/4);
  for(let i=0;i<out.length;i++)out[i]=dv.getFloat32(i*4,true);
  return out;
}

function makeCfg(labels,N,inputDim,dims,rep,binary,quant=null) {
  const enc=new TextEncoder();
  const labelBytes=labels.map(s=>enc.encode(String(s)));
  for(const b of labelBytes)if(!b.length||b.length>31)throw new Error('NAI labels must be 1..31 UTF-8 bytes');
  const quantBytes=quant ? (28 + 8*(dims.length-1)) : 0;
  const total=20 + 2*dims.length + labelBytes.reduce((s,b)=>s+1+b.length,0) + quantBytes;
  const out=new Uint8Array(total),dv=new DataView(out.buffer);
  out.set([0x4e,0x41,0x49,0x34],0); // NAI4
  let o=4;
  const flags=(binary?FLAG_BINARY_CLASSIFIER:0)|(quant?FLAG_INT8:0)|FLAG_LPF5|
    (['quaternion','velocity','velocity+quaternion','accel+roll+pitch'].includes(rep)?FLAG_CONT_ORIENTATION:0);
  const vals=[4,flags,N,SAMPLE_RATE_HZ,dims.length-1,labels.length,inputDim,REP[rep].code];
  vals.forEach(v=>{dv.setUint16(o,v,true);o+=2;});
  dims.forEach(v=>{dv.setUint16(o,v,true);o+=2;});
  for(const b of labelBytes){out[o++]=b.length;out.set(b,o);o+=b.length;}
  if(quant){
    out.set(new TextEncoder().encode(QCFG_MAGIC),o);o+=4;
    dv.setUint16(o,quant.strideSamples,true);o+=2;
    dv.setUint16(o,0,true);o+=2;
    dv.setFloat32(o,quant.thresholdMin,true);o+=4;
    dv.setFloat32(o,quant.thresholdMax,true);o+=4;
    dv.setFloat32(o,quant.confidenceThreshold,true);o+=4;
    dv.setFloat32(o,quant.inputScale,true);o+=4;
    dv.setInt32(o,quant.inputZeroPoint,true);o+=4;
    quant.layers.forEach(q=>{
      dv.setFloat32(o,q.outputScale,true);o+=4;
      dv.setInt32(o,q.outputZeroPoint,true);o+=4;
    });
  }
  return out;
}

async function extractTfWeightsOI(model) {
  const layers=[];
  for(const layer of model.layers) {
    const wb=layer.getWeights();
    if(wb.length!==2)throw new Error('Expected Dense layer kernel+bias');
    const [kernel,bias]=wb;
    const ks=kernel.shape; // [I,O]
    const I=ks[0],O=ks[1];
    const kd=await kernel.data(),bd=await bias.data();
    const oi=new Float32Array(O*I);
    for(let i=0;i<I;i++)for(let o=0;o<O;o++)oi[o*I+i]=kd[i*O+o];
    layers.push({I,O,w:oi,b:Float32Array.from(bd)});
  }
  return layers;
}

function roundEven(x) {
  if(!Number.isFinite(x)) return 0;
  const f=Math.floor(x), r=x-f;
  if(r<0.5)return f;
  if(r>0.5)return f+1;
  return (f%2===0)?f:f+1;
}
function clampI8(x){return x>127?127:(x<-128?-128:x);}
function clampI32(x){return x>2147483647?2147483647:(x<-2147483648?-2147483648:Math.trunc(x));}
function qFloat(x,scale,zp){return clampI8(roundEven(x/scale)+zp);}

// Mirrors noodle_quantize_multiplier() in noodle_int8_math.cpp.
function quantizeMultiplier(realMultiplier) {
  if(!(realMultiplier>=0)||!Number.isFinite(realMultiplier))throw new Error('Invalid quantized multiplier');
  if(realMultiplier===0)return {multiplier:0,shift:0};
  let exp=Math.floor(Math.log2(realMultiplier))+1;
  let q=realMultiplier/Math.pow(2,exp); // [0.5,1)
  let qFixed=Math.round(q*2147483648);
  if(qFixed===2147483648){qFixed/=2;exp+=1;}
  if(exp<-31)return {multiplier:0,shift:0};
  if(exp>30)throw new Error('Quantized multiplier exponent overflow');
  return {multiplier:qFixed,shift:exp};
}
function satRoundHighMul(a,b) {
  const ab=BigInt(Math.trunc(a))*BigInt(Math.trunc(b));
  const one=1n<<30n;
  const nudge=ab>=0n?one:(1n-one);
  return Number((ab+nudge)/(1n<<31n));
}
function roundingDivideByPot(x,exp) {
  if(exp<=0)return Math.trunc(x);
  if(exp>31)return x<0?-1:0;
  const xb=BigInt(Math.trunc(x));
  const mask=(1n<<BigInt(exp))-1n;
  const remainder=xb&mask;
  const threshold=(mask>>1n)+(x<0?1n:0n);
  return Number((xb>>BigInt(exp))+(remainder>threshold?1n:0n));
}
function multiplyByQuantizedMultiplier(x,multiplier,shift) {
  const left=shift>0?shift:0, right=shift>0?0:-shift;
  let shifted;
  if(left>=31)shifted=x>=0?2147483647:-2147483648;
  else shifted=clampI32(Math.trunc(x)*Math.pow(2,left));
  const high=satRoundHighMul(shifted,multiplier);
  return roundingDivideByPot(high,right);
}
function applyRequant(acc,mult,shift,zp,amin=-128,amax=127) {
  let q=multiplyByQuantizedMultiplier(acc,mult,shift)+zp;
  if(q<amin)q=amin;if(q>amax)q=amax;
  return clampI8(q);
}


function maxAbsRows(rows){
  let m=0;
  for(const r of rows)for(let i=0;i<r.length;i++){const a=Math.abs(r[i]);if(a>m)m=a;}
  return m;
}
function maxRows(rows){
  let m=0;
  for(const r of rows)for(let i=0;i<r.length;i++)if(r[i]>m)m=r[i];
  return m;
}
function denseFloatRows(rows,layer,relu){
  const out=new Array(rows.length);
  for(let n=0;n<rows.length;n++){
    const a=rows[n],z=new Float32Array(layer.O);
    for(let o=0;o<layer.O;o++){
      let s=layer.b[o],base=o*layer.I;
      for(let i=0;i<layer.I;i++)s+=layer.w[base+i]*a[i];
      z[o]=relu&&s<0?0:s;
    }
    out[n]=z;
  }
  return out;
}

function calibrateActivationQParams(floatLayers,calibrationRows){
  if(!calibrationRows.length)throw new Error('INT8 calibration needs training rows');
  const activations=[];
  const inMax=maxAbsRows(calibrationRows);
  activations.push({scale:Math.max(inMax/127,1e-12),zeroPoint:0});
  let rows=calibrationRows;
  for(let li=0;li<floatLayers.length;li++){
    const hidden=li<floatLayers.length-1;
    rows=denseFloatRows(rows,floatLayers[li],hidden);
    if(hidden){
      const hi=maxRows(rows);
      activations.push({scale:Math.max(hi/255,1e-12),zeroPoint:-128});
    }else{
      const ma=maxAbsRows(rows);
      activations.push({scale:Math.max(ma/127,1e-12),zeroPoint:0});
    }
  }
  return activations;
}

function quantizeDenseLayers(floatLayers,activations){
  return floatLayers.map((l,li)=>{
    const qW=new Int8Array(l.O*l.I),qB=new Int32Array(l.O),mult=new Int32Array(l.O),shift=new Int32Array(l.O);
    const weightScale=new Float64Array(l.O);
    const inScale=activations[li].scale,outScale=activations[li+1].scale;
    for(let o=0;o<l.O;o++){
      let ma=0,base=o*l.I;
      for(let i=0;i<l.I;i++){const a=Math.abs(l.w[base+i]);if(a>ma)ma=a;}
      const ws=Math.max(ma/127,1e-12);weightScale[o]=ws;
      for(let i=0;i<l.I;i++)qW[base+i]=clampI8(roundEven(l.w[base+i]/ws));
      qB[o]=clampI32(roundEven(l.b[o]/(inScale*ws)));
      const qm=quantizeMultiplier(inScale*ws/outScale);
      mult[o]=qm.multiplier;shift[o]=qm.shift;
    }
    return {I:l.I,O:l.O,w:qW,b:qB,mult,shift,
      inputScale:activations[li].scale,inputZeroPoint:activations[li].zeroPoint,
      outputScale:activations[li+1].scale,outputZeroPoint:activations[li+1].zeroPoint};
  });
}

function noodleBinaryProbability(q,scale,zp){
  const x=(q-zp)*scale;
  const y=x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const pi=Math.max(0,Math.min(255,Math.round(y*256)));
  return pi/256;
}
function noodleSoftmaxProbabilities(qrow,scale){
  let maxq=qrow[0];for(let i=1;i<qrow.length;i++)if(qrow[i]>maxq)maxq=qrow[i];
  const lut=new Uint16Array(256);
  for(let d=0;d<256;d++)lut[d]=Math.max(0,Math.min(32767,Math.round(Math.exp(-d*scale)*32767)));
  let sum=0;const e=new Uint16Array(qrow.length);
  for(let i=0;i<qrow.length;i++){const d=(maxq-qrow[i])&255;e[i]=lut[d];sum+=e[i];}
  const p=new Array(qrow.length);
  for(let i=0;i<qrow.length;i++){let pi=Math.floor((e[i]*256+Math.floor(sum/2))/sum);if(pi>255)pi=255;p[i]=pi/256;}
  return p;
}

function predictQuantized(qmodel,standardizedRows,nClasses){
  const probs=[],pred=[];
  for(const row of standardizedRows){
    let a=new Int16Array(row.length);
    const aq=qmodel.activations[0];
    for(let i=0;i<row.length;i++)a[i]=qFloat(row[i],aq.scale,aq.zeroPoint);
    qmodel.layers.forEach((l,li)=>{
      const z=new Int16Array(l.O),hidden=li<qmodel.layers.length-1;
      const amin=hidden?l.outputZeroPoint:-128;
      for(let o=0;o<l.O;o++){
        let acc=l.b[o],base=o*l.I;
        for(let i=0;i<l.I;i++)acc+=(a[i]-l.inputZeroPoint)*l.w[base+i];
        z[o]=applyRequant(clampI32(acc),l.mult[o],l.shift[o],l.outputZeroPoint,amin,127);
      }
      a=z;
    });
    let p;
    const final=qmodel.activations[qmodel.activations.length-1];
    if(nClasses===2){const p1=noodleBinaryProbability(a[0],final.scale,final.zeroPoint);p=[1-p1,p1];}
    else p=noodleSoftmaxProbabilities(a,final.scale);
    probs.push(p);pred.push(argmaxRow(p));
  }
  return {probs,pred};
}

function sweepConfidenceThreshold(probs,y,{min=0.30,max=0.90,step=0.01,minCoverage=0.90}={}){
  const rows=[];
  const pred=probs.map(argmaxRow),conf=probs.map(p=>Math.max(...p));
  const n=y.length;
  for(let k=0;;k++){
    const t=Number((min+k*step).toFixed(6));if(t>max+1e-9)break;
    let accepted=0,correct=0;
    for(let i=0;i<n;i++)if(conf[i]>=t){accepted++;if(pred[i]===y[i])correct++;}
    rows.push({threshold:t,coverage:accepted/n,acceptedAccuracy:accepted?correct/accepted:0,accepted,correct});
  }
  const eligible=rows.filter(r=>r.coverage+1e-12>=minCoverage&&r.accepted>0);
  const pool=eligible.length?eligible:rows.filter(r=>r.accepted>0);
  let selected=pool[0];
  for(const r of pool.slice(1)){
    if(r.acceptedAccuracy>selected.acceptedAccuracy+1e-12 ||
       (Math.abs(r.acceptedAccuracy-selected.acceptedAccuracy)<=1e-12 && r.threshold>selected.threshold))selected=r;
  }
  return {rows,selected,min,max,step,minCoverage};
}


async function exportNai4(model,scaler,labels,N,rep) {
  const layers=await extractTfWeightsOI(model);
  const dims=[layers[0].I,...layers.map(l=>l.O)];
  const binary=labels.length===2;
  const files={
    'cfg.bin':makeCfg(labels,N,dims[0],dims,rep,binary),
    'mean.bin':float32LE(scaler.mean),
    'scale.bin':float32LE(scaler.scale),
  };
  layers.forEach((l,i)=>{
    files[`w${String(i).padStart(2,'0')}.bin`]=float32LE(l.w);
    files[`b${String(i).padStart(2,'0')}.bin`]=float32LE(l.b);
  });
  const total=Object.values(files).reduce((s,b)=>s+b.byteLength,0);
  if(total>MAX_PACKAGE_BYTES)throw new Error(`Raw NAI4 package ${(total/1024).toFixed(1)} KiB exceeds 550 KiB`);
  const zip=new JSZip();
  Object.entries(files).forEach(([name,bytes])=>zip.file(name,bytes));
  const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE'});
  return {files,total,blob,layers,dims,binary,rep,labels,N,scaler};
}

function int32LE(values) {
  const out=new ArrayBuffer(values.length*4),dv=new DataView(out);
  for(let i=0;i<values.length;i++)dv.setInt32(i*4,Number(values[i]),true);
  return new Uint8Array(out);
}
function readInt32LE(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), out=new Int32Array(bytes.byteLength/4);
  for(let i=0;i<out.length;i++)out[i]=dv.getInt32(i*4,true);
  return out;
}
function int8Bytes(values) {
  const out=new Uint8Array(values.length);
  for(let i=0;i<values.length;i++)out[i]=Number(values[i])&0xff;
  return out;
}
function readInt8(bytes) {
  const out=new Int8Array(bytes.length);
  for(let i=0;i<bytes.length;i++)out[i]=bytes[i]>127?bytes[i]-256:bytes[i];
  return out;
}


// Mode 1: deterministic PTQ from the training split. Validation is evaluation-only.
// Keeps BOOT-triggered gesture policy and existing FP32 package unchanged.
function packedMetadataLE(bias, multiplier, shift) {
  if (bias.length !== multiplier.length || bias.length !== shift.length)
    throw new Error('Packed FCN metadata arrays must have equal output widths');
  const bytes = new Uint8Array(bias.length * 12);
  const dv = new DataView(bytes.buffer);
  for (let o = 0; o < bias.length; ++o) {
    dv.setInt32(o*12,     bias[o], true);
    dv.setInt32(o*12 + 4, multiplier[o], true);
    dv.setInt32(o*12 + 8, shift[o], true);
  }
  return bytes;
}

async function exportNai4Int8(model,scaler,labels,N,rep,calibrationRows,validationRows,validationY) {
  const floatLayers=await extractTfWeightsOI(model);
  if(floatLayers.length===0 || floatLayers.length>8)throw new Error('Expected 1..8 Dense layers');
  const dims=[floatLayers[0].I,...floatLayers.map(l=>l.O)];
  const binary=labels.length===2;
  const activations=calibrateActivationQParams(floatLayers,calibrationRows);
  const layers=quantizeDenseLayers(floatLayers,activations);
  const qmodel={activations,layers};
  const results=predictQuantized(qmodel,validationRows,labels.length);
  const validationAccuracy=accuracy(results.pred,Array.from(validationY));
  // Mode 1 is explicitly BOOT-delimited, not rolling-window-based. A fixed
  // confidence threshold here is metadata for future target implementation,
  // not validation-tuned rejection behavior.
  const quant={strideSamples:0,thresholdMin:0,thresholdMax:1,confidenceThreshold:0.5,
    inputScale:activations[0].scale,inputZeroPoint:activations[0].zeroPoint,
    layers:layers.map(l=>({outputScale:l.outputScale,outputZeroPoint:l.outputZeroPoint}))};
  const files={'cfg.bin':makeCfg(labels,N,dims[0],dims,rep,binary,quant),
    'mean.bin':float32LE(scaler.mean),'scale.bin':float32LE(scaler.scale)};
  layers.forEach((l,i)=>{
    const n=String(i).padStart(2,'0');
    files[`w${n}.bin`]=int8Bytes(l.w);
    files[`p${n}.bin`]=packedMetadataLE(l.b,l.mult,l.shift);
  });
  const total=Object.values(files).reduce((s,b)=>s+b.byteLength,0);
  if(total>MAX_PACKAGE_BYTES)throw new Error(`Raw NAI4 INT8 package ${(total/1024).toFixed(1)} KiB exceeds 550 KiB`);
  const zip=new JSZip();Object.entries(files).forEach(([name,bytes])=>zip.file(name,bytes));
  const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE'});
  return {files,total,blob,dims,binary,rep,labels,N,scaler,int8:true,quant,
    qmodel,validationAccuracy,validationPredictions:results.pred,validationProbabilities:results.probs};
}

function parseCfg(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const magic=String.fromCharCode(...bytes.slice(0,4));
  let o=4;
  const v=[];for(let i=0;i<8;i++){v.push(dv.getUint16(o,true));o+=2;}
  const [version,flags,N,rate,nLayers,nClasses,inputDim,reserved]=v;
  if(magic!=='NAI4'||version!==4)throw new Error('NAI4 v4 model required');
  if((flags&FLAG_LPF5)===0 || (flags&~(FLAG_BINARY_CLASSIFIER|FLAG_INT8|FLAG_LPF5|FLAG_CONT_ORIENTATION)))
    throw new Error('Incompatible preprocessing: a new 5 Hz LPF model is required. Retrain/export using this web version.');
  const rep=CODE_TO_REP[reserved]||null;
  if(!rep)throw new Error(`Unsupported representation code ${reserved}; retrain with the new input selection.`);
  if(rep==='accel+roll+pitch' && !(flags&FLAG_CONT_ORIENTATION))throw new Error('Continuous orientation flag is required for accel+roll+pitch');
  const dims=[];for(let i=0;i<nLayers+1;i++){dims.push(dv.getUint16(o,true));o+=2;}
  const dec=new TextDecoder('utf-8'),labels=[];
  for(let i=0;i<nClasses;i++){const n=bytes[o++];labels.push(dec.decode(bytes.slice(o,o+n)));o+=n;}
  const int8=!!(flags&FLAG_INT8);let quant=null;
  if(int8){
    if(o+28+8*nLayers>bytes.length)throw new Error('Truncated NAI4 INT8 config');
    const qmagic=new TextDecoder().decode(bytes.slice(o,o+4));o+=4;
    if(qmagic!==QCFG_MAGIC)throw new Error(`Unknown INT8 config ${qmagic}`);
    const strideSamples=dv.getUint16(o,true);o+=2;o+=2;
    const thresholdMin=dv.getFloat32(o,true);o+=4;
    const thresholdMax=dv.getFloat32(o,true);o+=4;
    const confidenceThreshold=dv.getFloat32(o,true);o+=4;
    const inputScale=dv.getFloat32(o,true);o+=4;
    const inputZeroPoint=dv.getInt32(o,true);o+=4;
    const layers=[];for(let i=0;i<nLayers;i++){
      const outputScale=dv.getFloat32(o,true);o+=4;
      const outputZeroPoint=dv.getInt32(o,true);o+=4;
      layers.push({outputScale,outputZeroPoint});
    }
    quant={strideSamples,thresholdMin,thresholdMax,confidenceThreshold,inputScale,inputZeroPoint,layers};
  }
  if(o!==bytes.length)throw new Error(`Unexpected cfg.bin trailing bytes (${bytes.length-o})`);
  return {magic,version,flags,N,rate,nLayers,nClasses,inputDim,rep,dims,labels,binary:!!(flags&FLAG_BINARY_CLASSIFIER),int8,quant};
}

async function loadNai(file) {
  const zip=await JSZip.loadAsync(await file.arrayBuffer());
  const get=async name=>{
    const e=zip.file(name);if(!e)throw new Error(`NAI missing ${name}`);
    return new Uint8Array(await e.async('arraybuffer'));
  };
  const cfg=parseCfg(await get('cfg.bin'));
  if(cfg.int8)throw new Error('This INT8 package requires the INT8 Noodle runtime; FP32 browser predictor cannot load it.');
  const mean=readFloat32LE(await get('mean.bin')),scale=readFloat32LE(await get('scale.bin'));
  const layers=[];
  for(let i=0;i<cfg.nLayers;i++){
    const I=cfg.dims[i],O=cfg.dims[i+1];
    const w=readFloat32LE(await get(`w${String(i).padStart(2,'0')}.bin`));
    const b=readFloat32LE(await get(`b${String(i).padStart(2,'0')}.bin`));
    if(w.length!==I*O||b.length!==O)throw new Error(`Layer ${i} byte count mismatch`);
    layers.push({I,O,w,b});
  }
  return {...cfg,mean,scale,layers,fileName:file.name};
}

function predictNai(nai, rows) {
  const probs=[],pred=[];
  for(const raw of rows){
    let a=new Float64Array(nai.inputDim);
    for(let i=0;i<a.length;i++)a[i]=(raw[i]-nai.mean[i])/nai.scale[i];
    nai.layers.forEach((l,li)=>{
      const z=new Float64Array(l.O);
      for(let o=0;o<l.O;o++){
        let s=l.b[o],base=o*l.I;
        for(let i=0;i<l.I;i++)s+=l.w[base+i]*a[i];
        z[o]=s;
      }
      if(li<nai.layers.length-1){for(let o=0;o<z.length;o++)if(z[o]<0)z[o]=0;}
      a=z;
    });
    let p;
    if(nai.binary){
      const p1=a[0]>=0?1/(1+Math.exp(-a[0])):Math.exp(a[0])/(1+Math.exp(a[0]));
      p=[1-p1,p1];
    }else{
      const m=Math.max(...a),e=Array.from(a,v=>Math.exp(v-m)),s=e.reduce((x,y)=>x+y,0);p=e.map(v=>v/s);
    }
    probs.push(p);pred.push(argmaxRow(p));
  }
  return {probs,pred};
}


// ---------------------------------------------------------------------------
// NPY / NPZ writer for Python-compatible NAI4 datasets
// ---------------------------------------------------------------------------

function makeNpyHeader(descr, shape) {
  const shapeText = shape.length === 0 ? '()' : `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeText}, }`;
  // NPY v1: magic(6)+version(2)+header_len(2)+header must align to 16 bytes.
  const preamble = 10;
  let pad = 16 - ((preamble + header.length + 1) % 16);
  if (pad === 16) pad = 0;
  header += ' '.repeat(pad) + '\n';
  const hb = new TextEncoder().encode(header);
  if (hb.length > 65535) throw new Error('NPY header too large');
  const out = new Uint8Array(10 + hb.length);
  out.set([0x93,0x4e,0x55,0x4d,0x50,0x59,0x01,0x00], 0);
  new DataView(out.buffer).setUint16(8, hb.length, true);
  out.set(hb, 10);
  return out;
}

function concatBytes(a,b) {
  const out = new Uint8Array(a.length+b.length);
  out.set(a,0); out.set(b,a.length); return out;
}

function npyF32(values, shape) {
  const header = makeNpyHeader('<f4', shape);
  const payload = new Uint8Array(values.length*4);
  const dv = new DataView(payload.buffer);
  for (let i=0;i<values.length;i++) dv.setFloat32(i*4, Number(values[i]), true);
  return concatBytes(header,payload);
}

function npyI32(values, shape) {
  const arr = Array.from(values, Number);
  const header = makeNpyHeader('<i4', shape);
  const payload = new Uint8Array(arr.length*4);
  const dv = new DataView(payload.buffer);
  for (let i=0;i<arr.length;i++) dv.setInt32(i*4, arr[i], true);
  return concatBytes(header,payload);
}

function npyUnicode(values, shape) {
  const arr = Array.from(values, v=>String(v));
  const maxChars = Math.max(1, ...arr.map(s=>Array.from(s).length));
  const header = makeNpyHeader(`<U${maxChars}`, shape);
  const payload = new Uint8Array(arr.length*maxChars*4);
  const dv = new DataView(payload.buffer);
  let off=0;
  for (const s of arr) {
    const cps = Array.from(s, ch=>ch.codePointAt(0));
    for (let j=0;j<maxChars;j++) { dv.setUint32(off, cps[j] || 0, true); off += 4; }
  }
  return concatBytes(header,payload);
}

function flattenRawRows(rawRows) {
  let total=0;
  for (const r of rawRows) total += r.length;
  const out = new Float32Array(total*6);
  const offsets = new Int32Array(rawRows.length+1);
  let row=0;
  rawRows.forEach((r,i)=>{
    offsets[i]=row;
    out.set(r.data, row*6);
    row += r.length;
  });
  offsets[rawRows.length]=row;
  return {data:out, offsets, totalRows:row};
}

async function buildDatasetNpzBlob(ds) {
  if (!ds || !ds.Xrows?.length) throw new Error('Dataset is empty');
  const zip = new JSZip();
  const n=ds.Xrows.length, D=ds.N*6;
  const X=flattenRows(ds.Xrows);
  zip.file('X.npy', npyF32(X,[n,D]));
  zip.file('y.npy', npyI32(ds.y,[n]));
  zip.file('labels.npy', npyUnicode(ds.labels,[ds.labels.length]));
  zip.file('normalized_length.npy', npyI32([ds.N],[]));
  zip.file('window_length.npy', npyI32([ds.N],[]));
  zip.file('raw_lengths.npy', npyI32(ds.rawLengths || Array(n).fill(ds.N),[n]));
  zip.file('durations_ms.npy', npyI32(ds.durationsMs || Array(n).fill(0),[n]));
  zip.file('sample_rate_hz.npy', npyI32([ds.sampleRate || SAMPLE_RATE_HZ],[]));
  zip.file('channels.npy', npyUnicode(['ax','ay','az','gx','gy','gz'],[6]));
  zip.file('channel_count.npy', npyI32([6],[]));
  zip.file('preprocess.npy', npyUnicode(['nai4_raw6_plus_continuous_q_v2'],[]));
  if (ds.rawRows && ds.rawRows.length === n) {
    const withQuat=ds.rawRows.filter(r=>r.quat && r.quat.length===r.length*4).length;
    if(withQuat!==0 && withQuat!==n)
      throw new Error('Dataset mixes old raw-only and new continuous-orientation recordings. Export them separately.');
    const raw=flattenRawRows(ds.rawRows);
    zip.file('raw_offsets.npy', npyI32(raw.offsets,[n+1]));
    zip.file('raw_data.npy', npyF32(raw.data,[raw.totalRows,6]));
    if(withQuat===n){
      const qs=new Float32Array(raw.totalRows*4);let off=0;
      for(const r of ds.rawRows){qs.set(r.quat,off);off+=r.length*4;}
      zip.file('quat_data.npy',npyF32(qs,[raw.totalRows,4]));
    }
  }
  return await zip.generateAsync({type:'blob', compression:'DEFLATE'});
}

function normalizeRawSixAxis(rawObj, targetN) {
  return centerColumns(resample(rawObj.data, rawObj.length, 6, targetN), targetN, 6);
}


window.NAI = Object.freeze({
  SAMPLE_RATE_HZ, MADGWICK_BETA, GRAVITY_MPS2, MAX_PACKAGE_BYTES,
  REP, CODE_TO_REP,
  loadNpz, parseDatasetArrays, buildDatasetNpzBlob,
  resample, centerColumns, normalizeRawSixAxis,
  sklearnStratifiedSplit, buildRepresentationDataset,
  fitScaler, configureRepresentationScaler, standardizeRows, flattenRows,
  makeModel, predictTf, accuracy,
  exportNai4, exportNai4Int8, loadNai, predictNai, predictQuantized,
  _internals:Object.freeze({makeNpyHeader,npyF32,npyI32,npyUnicode,parseNpy,MT19937,makeCfg,parseCfg,filterRaw6Lpf5}),
});
