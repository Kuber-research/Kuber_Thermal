#!/usr/bin/env python3
"""Build the interactive Ground-Truth-vs-Prediction viewer (self-contained).

Inlines three.js + OrbitControls + the exported case data (GT + Kuber prediction +
geometry) into one HTML file: two synced 3D views (CFD ground truth | Kuber
prediction), the solid geometry as a semi-transparent body, the fluid field as a
point cloud colored by temperature or velocity. Writes site/demo.html and, if a
path arg is given, an artifact-body copy.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(ROOT.replace("/thermabench", "/cfd_thermal_mvp"), "demo", "static", "vendor")
SP = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
DATA_JSON = sys.argv[3] if len(sys.argv) > 3 else os.path.join(SP, "viewer_data.json")

three = open(os.path.join(VENDOR, "three.min.js")).read()
orbit = open(os.path.join(VENDOR, "OrbitControls.js")).read()
data = open(DATA_JSON).read()

STYLE = r"""
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;height:100%;background:#0B0E13;color:#E7ECF2;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
#app{display:flex;flex-direction:column;height:100vh;min-height:520px}
header{padding:14px 20px 10px;border-bottom:1px solid #1c2530}
h1{font-family:Georgia,"Times New Roman",serif;font-weight:600;font-size:1.28rem;margin:0 0 2px}
.hint{color:#8A97A6;font-size:.82rem}
.bar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:10px 20px;border-bottom:1px solid #1c2530}
.grp{display:flex;gap:6px;align-items:center}
.grp>span{color:#8A97A6;font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;margin-right:2px}
button{background:#151B24;color:#C7D0DA;border:1px solid #263140;border-radius:999px;
  padding:.42em .95em;font-size:.86rem;cursor:pointer;transition:.15s}
button:hover{border-color:#3a4b5e;color:#fff}
button.on{background:#1F4E79;border-color:#2f6ea8;color:#fff}
.metrics{margin-left:auto;display:flex;gap:18px;font-size:.86rem;color:#C7D0DA;font-variant-numeric:tabular-nums}
.metrics b{color:#fff;font-weight:600}
#stage{position:relative;flex:1;min-height:0}
#stage canvas{display:block}
.vlabel{position:absolute;top:12px;font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;
  color:#C7D0DA;background:rgba(11,14,19,.6);padding:4px 10px;border:1px solid #263140;border-radius:6px;pointer-events:none}
#lg{left:14px}#lp{right:14px}#lp b{color:#7CB2E8}
#cbar{position:absolute;left:50%;transform:translateX(-50%);bottom:14px;display:flex;align-items:center;gap:10px;
  background:rgba(11,14,19,.72);border:1px solid #263140;border-radius:8px;padding:8px 12px}
#cbar .ramp{width:220px;height:12px;border-radius:3px}
#cbar span{font-size:.76rem;color:#C7D0DA;font-variant-numeric:tabular-nums}
#cbar .lab{color:#8A97A6}
@media (max-width:640px){#cbar .ramp{width:130px}.metrics{width:100%;margin:6px 0 0}}
"""

VIEWER_JS = r"""
const D = window.KUBER_DATA;
const LUTS = {
  inferno:[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]],
  viridis:[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]
};
function cmap(name,t){t=Math.max(0,Math.min(1,t));const L=LUTS[name];const x=t*(L.length-1);
  const i=Math.floor(x),f=x-i,a=L[i],b=L[Math.min(i+1,L.length-1)];
  return [(a[0]+(b[0]-a[0])*f)/255,(a[1]+(b[1]-a[1])*f)/255,(a[2]+(b[2]-a[2])*f)/255];}
function rampCss(name){const L=LUTS[name];return 'linear-gradient(90deg,'+L.map((c,i)=>
  'rgb('+c[0]+','+c[1]+','+c[2]+') '+Math.round(100*i/(L.length-1))+'%').join(',')+')';}

let scene,camera,renderer,controls,root=null;
let cur={cse:'heatsink',fld:'T'};

function frameParams(coords){let mn=[1e9,1e9,1e9],mx=[-1e9,-1e9,-1e9];
  for(const p of coords)for(let k=0;k<3;k++){if(p[k]<mn[k])mn[k]=p[k];if(p[k]>mx[k])mx[k]=p[k];}
  const c=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  const s=Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])||1;return {c,s:2/s};}
function tx(p,fp){return [(p[0]-fp.c[0])*fp.s,(p[1]-fp.c[1])*fp.s,(p[2]-fp.c[2])*fp.s];}

function spriteLabel(text,accent){
  const cv=document.createElement('canvas');cv.width=512;cv.height=128;const g=cv.getContext('2d');
  g.fillStyle=accent?'#7CB2E8':'#E7ECF2';g.font='bold 64px Georgia, serif';g.textAlign='center';g.textBaseline='middle';
  g.fillText(text,256,64);
  const tex=new THREE.CanvasTexture(cv);tex.minFilter=THREE.LinearFilter;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));
  sp.scale.set(1.05,0.27,1);return sp;}

function makeSide(coords,vals,fp,xoff,boxes,vmin,vmax,isHS,label,accent){
  const grp=new THREE.Group();grp.position.x=xoff;
  const N=coords.length,pos=new Float32Array(N*3),col=new Float32Array(N*3);
  const cn=cur.fld==='T'?'inferno':'viridis';
  for(let i=0;i<N;i++){const p=tx(coords[i],fp);pos[3*i]=p[0];pos[3*i+1]=p[1];pos[3*i+2]=p[2];
    const t=(vals[i]-vmin)/(vmax-vmin+1e-9),rgb=cmap(cn,t);col[3*i]=rgb[0];col[3*i+1]=rgb[1];col[3*i+2]=rgb[2];}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  grp.add(new THREE.Points(geo,new THREE.PointsMaterial({size:0.027,vertexColors:true,sizeAttenuation:true,transparent:true,opacity:0.92})));
  let topz=-1e9;
  for(const bx of boxes){const a=tx(bx[0],fp),b=tx(bx[1],fp);
    const w=Math.abs(b[0]-a[0])||0.002,h=Math.abs(b[1]-a[1])||0.002,d=Math.abs(b[2]-a[2])||0.002;
    const bg=new THREE.BoxGeometry(w,h,d);
    const m=new THREE.Mesh(bg,new THREE.MeshStandardMaterial({color:0xB8C0CC,metalness:0.35,roughness:0.5,transparent:true,opacity:isHS?0.22:0.05}));
    m.position.set((a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2);grp.add(m);
    const e=new THREE.LineSegments(new THREE.EdgesGeometry(bg),new THREE.LineBasicMaterial({color:0x8A97A6,transparent:true,opacity:0.45}));
    e.position.copy(m.position);grp.add(e);
    if(m.position.z+d/2>topz)topz=m.position.z+d/2;}
  const lab=spriteLabel(label,accent);lab.position.set(0,0,Math.max(topz,1)+0.45);grp.add(lab);
  return grp;}

function build(){
  if(root){scene.remove(root);root.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose&&o.material.dispose();});}
  root=new THREE.Group();
  const data=D[cur.cse],coords=data.coords,fp=frameParams(coords);
  const fld=data.fields[cur.fld==='T'?'T':'velocity'],gt=fld.gt,pr=fld.pred;
  const all=gt.concat(pr).slice().sort((a,b)=>a-b);
  const vmin=all[Math.floor(all.length*0.02)],vmax=all[Math.floor(all.length*0.98)];
  const isHS=data.device==='heatsink',off=1.6;
  root.add(makeSide(coords,gt,fp,-off,data.boxes,vmin,vmax,isHS,'Ground truth',false));
  root.add(makeSide(coords,pr,fp,+off,data.boxes,vmin,vmax,isHS,'Kuber',true));
  scene.add(root);
  const unit=cur.fld==='T'?'K':'m/s';
  document.querySelector('#cbar .ramp').style.background=rampCss(cur.fld==='T'?'inferno':'viridis');
  document.querySelector('#cbar .lo').textContent=vmin.toFixed(cur.fld==='T'?0:2);
  document.querySelector('#cbar .hi').textContent=vmax.toFixed(cur.fld==='T'?0:2);
  document.querySelector('#cbar .u').textContent=unit;
  const m=data.metrics;
  document.getElementById('metrics').innerHTML=
    'T&#8209;RMSE <b>'+m.T_rmse+' K</b> &nbsp;·&nbsp; peak&nbsp;T pred <b>'+m.peak_T+'</b> / GT <b>'+m.peak_T_gt+' K</b> &nbsp;·&nbsp; '+data.n_points+' pts';
  document.getElementById('sub').textContent=data.title+'  —  '+data.sub;
}

function init(){
  const stage=document.getElementById('stage');
  renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setClearColor(0x0B0E13,1);
  stage.appendChild(renderer.domElement);
  scene=new THREE.Scene();scene.fog=new THREE.Fog(0x0B0E13,9,16);
  camera=new THREE.PerspectiveCamera(38,1,0.01,100);camera.up.set(0,0,1);camera.position.set(0.35,-4.4,1.9);
  controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=0.08;controls.target.set(0,0,0.1);
  scene.add(new THREE.AmbientLight(0xffffff,0.75));
  const dl=new THREE.DirectionalLight(0xffffff,0.55);dl.position.set(4,-5,7);scene.add(dl);
  build();resize();window.addEventListener('resize',resize);
  document.querySelectorAll('[data-cse]').forEach(b=>b.onclick=()=>{cur.cse=b.dataset.cse;setOn('cse');build();});
  document.querySelectorAll('[data-fld]').forEach(b=>b.onclick=()=>{cur.fld=b.dataset.fld;setOn('fld');build();});
  (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();
}
function setOn(kind){document.querySelectorAll('[data-'+kind+']').forEach(b=>b.classList.toggle('on',b.dataset[kind]===cur[kind]));}
function resize(){const s=document.getElementById('stage'),w=s.clientWidth,h=s.clientHeight;
  renderer.setSize(w,h,false);camera.aspect=w/h||1;camera.updateProjectionMatrix();}
window.addEventListener('DOMContentLoaded',init);
"""

BODY_TEMPLATE = """
<title>Kuber Demo</title>
<style>__STYLE__</style>
<div id="app">
  <header>
    <h1>Ground truth vs. Kuber prediction</h1>
    <div class="hint" id="sub">Drag to orbit · scroll to zoom. The solid is the device; points are the fluid, colored by field.</div>
  </header>
  <div class="bar">
    <div class="grp"><span>Case</span>
      <button class="on" data-cse="heatsink">Heatsink</button>
      <button data-cse="coldplate">Cold plate</button></div>
    <div class="grp"><span>Field</span>
      <button class="on" data-fld="T">Temperature</button>
      <button data-fld="velocity">Velocity</button></div>
    <div class="metrics" id="metrics"></div>
  </div>
  <div id="stage">
    <div class="vlabel" id="lg">CFD ground truth</div>
    <div class="vlabel" id="lp"><b>Kuber prediction</b></div>
    <div id="cbar"><span class="lab lo"></span><div class="ramp"></div><span class="lab hi"></span><span class="u"></span></div>
  </div>
</div>
<script>__THREE__</script>
<script>__ORBIT__</script>
<script>window.KUBER_DATA=__DATA__;</script>
<script>__JS__</script>
"""

body = (BODY_TEMPLATE
        .replace("__STYLE__", STYLE)
        .replace("__THREE__", three)
        .replace("__ORBIT__", orbit)
        .replace("__DATA__", data)
        .replace("__JS__", VIEWER_JS))

full = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        + body + "\n</html>\n")

os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
open(os.path.join(ROOT, "site", "demo.html"), "w").write(full)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(body)
print("wrote site/demo.html", f"({len(full)//1024} KB)")
