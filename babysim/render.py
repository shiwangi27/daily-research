"""Render BabySim into a polished soft-toy (claymation/3-D) comic page — a self-contained
canvas page (content-only HTML for the Artifact publisher). Bilateral body (two arms, two
legs) with depth, and real anti-phase gait for walking/crawling. Kinematics + RL motion are
driven by the learned joint angles from babysim/poses.json.

    python3 -m babysim.render        # reads babysim/poses.json -> babysim/comic.html
"""
from __future__ import annotations

import json
import os

HTML = r"""<title>BabySim — Learning to Move</title>
<style>
:root{
  --bg:#fdeede; --bg2:#f6d9c4; --ink:#4a3a52; --muted:#9a8aa0; --accent:#ff7a6b;
  --card:#fff6ef; --dark:0;
}
@media (prefers-color-scheme:dark){:root{--bg:#241f36;--bg2:#191527;--ink:#efe7f5;--muted:#a99fc2;--accent:#ff8f7e;--card:#2c2743;--dark:1}}
:root[data-theme="light"]{--bg:#fdeede;--bg2:#f6d9c4;--ink:#4a3a52;--muted:#9a8aa0;--accent:#ff7a6b;--card:#fff6ef;--dark:0}
:root[data-theme="dark"]{--bg:#241f36;--bg2:#191527;--ink:#efe7f5;--muted:#a99fc2;--accent:#ff8f7e;--card:#2c2743;--dark:1}
*{box-sizing:border-box}
.bs{--disp:"Baloo 2","Trebuchet MS","Comic Sans MS",system-ui,sans-serif;--body:system-ui,-apple-system,sans-serif;
  font-family:var(--body);color:var(--ink);background:var(--bg);min-height:100%;padding:clamp(14px,3.5vw,36px);line-height:1.5}
.bs .wrap{max-width:1000px;margin:0 auto}
.bs header{margin-bottom:16px}
.bs .kicker{font-family:var(--disp);text-transform:uppercase;letter-spacing:.2em;font-size:.72rem;color:var(--accent);font-weight:700}
.bs h1{font-family:var(--disp);font-weight:800;line-height:.95;font-size:clamp(2.3rem,7vw,4rem);margin:.08em 0 .12em;color:var(--accent)}
.bs .lede{max-width:60ch;color:var(--muted)}.bs .lede b{color:var(--ink)}
.bs .stage{position:relative;border-radius:26px;overflow:hidden;box-shadow:0 24px 46px -22px rgba(150,90,70,.55)}
.bs .stage canvas{display:block;width:100%;aspect-ratio:1000/580}
.bs .bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding:13px 18px;background:var(--card)}
.bs .wk{font-family:var(--disp);font-weight:800;background:var(--ink);color:var(--card);padding:2px 13px;border-radius:999px;font-variant-numeric:tabular-nums}
.bs .ttl{font-family:var(--disp);font-weight:800;font-size:1.2rem;flex:1;min-width:150px}
.bs .stars{letter-spacing:2px;color:var(--accent)}
.bs .status{font-family:var(--disp);font-weight:700;font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;padding:3px 10px;border-radius:999px}
.bs .status.master{background:var(--accent);color:#fff}.bs .status.learn{background:#7cc3c9;color:#fff}
.bs .cap{flex-basis:100%;color:var(--muted);font-size:.92rem}
.bs .controls{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin:14px 0 0}
.bs button.b{font-family:var(--disp);font-weight:700;cursor:pointer;border:0;border-radius:999px;padding:8px 15px;
  font-size:.88rem;background:var(--card);color:var(--ink);box-shadow:0 3px 8px -2px rgba(120,80,70,.4)}
.bs button.b.play{background:var(--accent);color:#fff}
.bs button.b[aria-current="true"]{background:var(--accent);color:#fff}
.bs button.b:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.bs .slider{display:flex;flex-direction:column;gap:3px;flex:1;min-width:240px;margin-top:14px}
.bs .slider .row{display:flex;align-items:center;gap:12px}
.bs .slider input{flex:1;accent-color:var(--accent);height:26px}
.bs .read{font-family:var(--disp);font-weight:800;font-size:.88rem;white-space:nowrap;min-width:158px}
.bs .read span{color:var(--accent)}
.bs .ticks{display:flex;justify-content:space-between;font-size:.62rem;color:var(--muted);font-variant-numeric:tabular-nums;padding:0 4px}
.bs .foot{margin-top:22px;color:var(--muted);font-size:.83rem;line-height:1.6}
.bs .foot code{font-family:ui-monospace,Menlo,monospace;background:var(--card);padding:1px 6px;border-radius:5px;color:var(--ink)}
</style>

<div class="bs"><div class="wrap">
  <header>
    <div class="kicker">A Reinforcement-Learning Comic</div>
    <h1>BabySim</h1>
    <p class="lede">A tiny 2-D baby teaches herself each motor milestone from scratch with
      <b>REINFORCE</b>. Pick a skill, scrub the weeks, or press play to watch the whole first year.</p>
  </header>
  <div class="stage"><canvas id="cv" width="1000" height="580" role="img" aria-label="Animated baby learning to move"></canvas>
    <div class="bar">
      <span class="wk" id="wk">Week 6</span><span class="ttl" id="ttl">Lifts head</span>
      <span class="stars" id="stars">★★★★</span><span class="status master" id="st">Mastered</span>
      <span class="cap" id="cap"></span>
    </div>
  </div>
  <div class="controls"><button class="b play" id="playAll" type="button">▶ Play the first year</button>
    <span id="skills" style="display:flex;flex-wrap:wrap;gap:9px"></span></div>
  <div class="slider"><div class="row">
    <input type="range" id="week" min="1" max="56" value="6" step="1" aria-label="Week">
    <span class="read">Week <span id="wkNum">6</span> · <span id="wkSkill">head</span></span></div>
    <div class="ticks" id="ticks"></div></div>
  <p class="foot">Every pose is a real training run — a policy-gradient (REINFORCE) controller over the
    baby's joint angles. Two limbs on each side swing in anti-phase for gait; between milestones the
    body is interpolated. Offline &amp; reproducible: <code>python3 -m babysim.sim</code> →
    <code>python3 -m babysim.render</code>.</p>
</div></div>

<script>
const DATA=__DATA__, L=DATA.L, M=DATA.milestones;
const reduce=matchMedia("(prefers-reduced-motion:reduce)").matches;
const lerp=(a,b,t)=>a+(b-a)*t, TAU=Math.PI*2;
const lerpAng=(A,B,t)=>{const o={};for(const k in A)o[k]=lerp(A[k],B[k]??A[k],t);return o;};
const NEWBORN={px:0,py:0.34,torso:1.5,head:0.55,shoulder:2.15,elbow:1.5,hip:1.6,knee:1.7};
const d=t=>[Math.sin(t),Math.cos(t)], ad=(p,v,s)=>[p[0]+v[0]*s,p[1]+v[1]*s];

function body(a){const pelvis=[a.px,a.py],chest=ad(pelvis,d(a.torso),L.torso),
  neck=ad(chest,d(a.torso),L.neck),head=ad(neck,d(a.torso+a.head),L.head);return {pelvis,chest,neck,head};}
function legPts(pelvis,hip,knee){const k=ad(pelvis,d(hip),L.thigh);return {knee:k,foot:ad(k,d(hip+knee),L.shin)};}
function armPts(chest,sh,el){const e=ad(chest,d(sh),L.uarm);return {elbow:e,hand:ad(e,d(sh+el),L.farm)};}

// bounds (single-limb mastered poses + newborn + toy) for a fixed, stable transform
function gbounds(){let xs=[],ys=[];[NEWBORN,...M.map(m=>m.angles)].forEach(a=>{const b=body(a),
  l=legPts(b.pelvis,a.hip,a.knee),r=armPts(b.chest,a.shoulder,a.elbow);
  [b.pelvis,b.chest,b.head,l.knee,l.foot,r.elbow,r.hand].forEach(p=>{xs.push(p[0]);ys.push(p[1]);});});
  xs.push(DATA.rattle[0]);ys.push(DATA.rattle[1]);
  // generous margins so the baby reads small in a big room
  return {minx:Math.min(...xs)-3.4,maxx:Math.max(...xs)+3.4,miny:-.7,maxy:Math.max(...ys)+2.4};}
const GB=gbounds();
function fitter(W,H,pad){const b=GB,s=Math.min((W-2*pad)/(b.maxx-b.minx),(H-2*pad)/(b.maxy-b.miny));
  const ox=pad+((W-2*pad)-s*(b.maxx-b.minx))/2,oy=pad+((H-2*pad)-s*(b.maxy-b.miny))/2;
  return {T:p=>[ox+s*(p[0]-b.minx),H-(oy+s*(p[1]-b.miny))],s,ground:H-(oy+s*(0-b.miny))};}

// build a bilateral rig (angles for near/far arm+leg) from a base pose + per-skill motion
function rigOf(a,key,p){
  let base=Object.assign({},a),scroll=0,legAmp=0,armSwing=0,ph=0,bat=0;
  const e=0.5-0.5*Math.cos(TAU*p), sw=Math.sin(TAU*p);
  if(key==="head")base.head=lerp(0.35,a.head,e);
  else if(key==="reach"){base.shoulder=lerp(a.shoulder,a.shoulder,e);bat=1;}
  else if(key==="sit"){base.torso=a.torso+0.16*sw;base.head=a.head+0.1*Math.sin(2*TAU*p);}
  else if(key==="crawl"){scroll=1;legAmp=0.5;armSwing=0.5;ph=TAU*p;base.head=a.head+0.05*sw;}
  else if(key==="stand"){base.py=lerp(a.py-0.55,a.py,e);base.knee=lerp(a.knee+1.15,a.knee,e);base.hip=lerp(a.hip-0.55,a.hip,e);base.torso=a.torso+0.05*sw;}
  else if(key==="walk"){scroll=1;legAmp=0.6;armSwing=0.32;ph=TAU*p;base.torso=a.torso+0.05*Math.sin(2*TAU*p);base.py=a.py+0.05*Math.abs(Math.sin(TAU*p));}
  const rg={px:base.px,py:base.py,torso:base.torso,head:base.head,legs:[],arms:[]};
  [1,0].forEach(near=>{const o=near?0:1; // o=0 near, o=1 far
    let hip=base.hip,knee=base.knee,sh=base.shoulder,el=base.elbow;
    if(legAmp){const a2=ph+(o?Math.PI:0);hip=base.hip+legAmp*Math.sin(a2);knee=base.knee+0.75*Math.max(0,Math.sin(a2+0.5));}
    else hip=base.hip+(o?-0.13:0.05);            // slight splay -> two legs read
    if(armSwing){const a2=ph+(o?0:Math.PI);sh=base.shoulder+armSwing*Math.sin(a2);}
    else if(bat){sh=base.shoulder+(o?-0.18:0.18)*sw;}
    else sh=base.shoulder+(o?0.14:-0.04);
    rg.legs.push({hip,knee,far:!!o});rg.arms.push({shoulder:sh,elbow:el,far:!!o});});
  return {rig:rg,scroll};
}
function staticRig(a){return rigOf(a,"_",0).rig;}
function rigPoints(rg,fit){const T=fit.T,b=body(rg);
  return {pelvis:T(b.pelvis),chest:T(b.chest),neck:T(b.neck),head:T(b.head),
    legs:rg.legs.map(l=>{const q=legPts(b.pelvis,l.hip,l.knee);return {knee:T(q.knee),foot:T(q.foot),far:l.far};}),
    arms:rg.arms.map(m=>{const q=armPts(b.chest,m.shoulder,m.elbow);return {elbow:T(q.elbow),hand:T(q.hand),far:m.far};}),
    rattle:T(DATA.rattle)};}

// ---------- soft-toy painter ----------
const cv=document.getElementById("cv"),ctx=cv.getContext("2d");
let floorX=0;
const PAL={skin:"#f7c9a2",skinHi:"#ffe6cf",skinFar:"#e7b38a",suit:"#ff7a6b",suitHi:"#ffb3a6",
  suitFar:"#d55f54",belly:"#7dd0d4",bootie:"#8a6cff",bootieFar:"#6d51e0",hair:"#764a34",toy:"#ffcf4d",cheek:"#ff9aa0"};
function blob(a,b,w,col,hi){const g=ctx.createLinearGradient(a[0]-w,a[1]-w,a[0]+w,a[1]+w);
  g.addColorStop(0,hi);g.addColorStop(.55,col);g.addColorStop(1,col);
  ctx.strokeStyle=g;ctx.lineWidth=w;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
function ball(c,r,col,hi){const g=ctx.createRadialGradient(c[0]-r*.35,c[1]-r*.42,r*.12,c[0],c[1],r);
  g.addColorStop(0,hi);g.addColorStop(1,col);ctx.fillStyle=g;ctx.beginPath();ctx.arc(c[0],c[1],r,0,TAU);ctx.fill();}

function drawScene(fit){const W=cv.width,H=cv.height,dark=getComputedStyle(document.querySelector(".bs")).getPropertyValue("--dark").trim()==="1";
  const g=ctx.createLinearGradient(0,0,0,H);
  if(dark){g.addColorStop(0,"#2b2542");g.addColorStop(1,"#211c33");}else{g.addColorStop(0,"#fdeede");g.addColorStop(1,"#ffe3cf");}
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  // window of soft light
  ctx.save();ctx.globalAlpha=dark?.10:.5;const wx=W*0.13,wy=H*0.12,ww=W*0.2,wh=H*0.34;
  const wg=ctx.createLinearGradient(0,wy,0,wy+wh);wg.addColorStop(0,dark?"#8fa6d6":"#fff6e0");wg.addColorStop(1,dark?"#6b7fb0":"#ffe6bd");
  ctx.fillStyle=wg;rr(wx,wy,ww,wh,18);ctx.fill();ctx.restore();
  // floor band + rug
  const gy=fit.ground;const fg=ctx.createLinearGradient(0,gy-10,0,H);
  fg.addColorStop(0,dark?"#3a3357":"#f3c9a6");fg.addColorStop(1,dark?"#2c2743":"#e6a87e");
  ctx.fillStyle=fg;ctx.fillRect(0,gy,W,H-gy);
  ctx.save();ctx.globalAlpha=dark?.5:.7;ctx.fillStyle=dark?"#48416b":"#ffd9b0";
  ctx.beginPath();ctx.ellipse(W*0.52,gy+ (H-gy)*0.42,W*0.42,(H-gy)*0.5,0,0,TAU);ctx.fill();ctx.restore();
  // scrolling floor dots (imply forward motion)
  ctx.save();ctx.globalAlpha=dark?.22:.30;ctx.fillStyle=dark?"#efe7f5":"#c98a5c";
  for(let x=(-floorX%60);x<W;x+=60)for(let j=0;j<3;j++){const yy=gy+22+j*((H-gy-30)/3);
    ctx.beginPath();ctx.arc(x,yy,3.2,0,TAU);ctx.fill();}ctx.restore();
}
function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}

function limbArm(P,a,near){const c=near?PAL.skin:PAL.skinFar,hi=near?PAL.skinHi:PAL.skin,su=near?PAL.suit:PAL.suitFar,suhi=near?PAL.suitHi:PAL.suit,s=P._s;
  blob(P.chest,a.elbow,s*(near?0.3:0.26),su,suhi);blob(a.elbow,a.hand,s*(near?0.27:0.23),c,hi);ball(a.hand,s*(near?0.17:0.15),c,hi);}
function limbLeg(P,l,near){const su=near?PAL.suit:PAL.suitFar,suhi=near?PAL.suitHi:PAL.suit,bo=near?PAL.bootie:PAL.bootieFar,s=P._s;
  blob(P.pelvis,l.knee,s*(near?0.34:0.29),su,suhi);blob(l.knee,l.foot,s*(near?0.3:0.26),su,suhi);ball(l.foot,s*(near?0.2:0.17),bo,"#c3b4ff");}

function drawBaby(rg,fit,opts){opts=opts||{};const P=rigPoints(rg,fit),s=fit.s;P._s=s;
  // contact shadow
  ctx.save();ctx.fillStyle="rgba(70,45,35,.22)";ctx.filter="blur(7px)";
  ctx.beginPath();ctx.ellipse((P.pelvis[0]+P.chest[0])/2,fit.ground+3,s*1.0,s*0.16,0,0,TAU);ctx.fill();ctx.restore();
  const far=i=>P.legs[i].far, farA=i=>P.arms[i].far;
  // FAR limbs (behind)
  P.arms.forEach((a,i)=>{if(a.far)limbArm(P,a,false);});
  P.legs.forEach((l,i)=>{if(l.far)limbLeg(P,l,false);});
  // torso
  ctx.save();ctx.shadowColor="rgba(50,30,20,.16)";ctx.shadowBlur=s*0.18;ctx.shadowOffsetY=s*0.06;
  blob(P.pelvis,P.chest,s*0.64,PAL.suit,PAL.suitHi);ctx.restore();
  ball([(P.pelvis[0]+P.chest[0])/2,(P.pelvis[1]+P.chest[1])/2],s*0.2,PAL.belly,"#c7efff");
  // NEAR limbs (front)
  P.legs.forEach(l=>{if(!l.far)limbLeg(P,l,true);});
  P.arms.forEach(a=>{if(!a.far)limbArm(P,a,true);});
  // toy (reach): a dangling rattle above
  if(opts.rattle){const r=P.rattle;ctx.strokeStyle="#caa24a";ctx.lineWidth=s*0.07;ctx.lineCap="round";
    ctx.beginPath();ctx.moveTo(r[0],r[1]-s*0.9);ctx.lineTo(r[0],r[1]);ctx.stroke();
    ball(r,s*0.26,PAL.toy,"#fff2c0");ball([r[0]-s*.08,r[1]-s*.08],s*0.06,"#ffffff","#fff");}
  // head — features drawn in a rotated local frame so the face can point up when supine
  const h=P.head,hr=s*0.6;ball(h,hr,PAL.skin,PAL.skinHi);
  ctx.save();ctx.translate(h[0],h[1]);ctx.rotate(opts.face||0);
  ctx.fillStyle=PAL.hair;ctx.beginPath();ctx.arc(0,-hr*0.5,hr*0.72,Math.PI,TAU);ctx.fill();
  ctx.beginPath();ctx.arc(hr*0.12,-hr*0.78,hr*0.26,0,TAU);ctx.fill();
  const ex=hr*0.17,ey=-hr*0.02;
  [-1,1].forEach(k=>{ball([ex+k*hr*0.3,ey],hr*0.15,"#fff","#fff");ctx.fillStyle="#33263a";
    ctx.beginPath();ctx.arc(ex+k*hr*0.3+hr*0.03,ey+hr*0.02,hr*0.075,0,TAU);ctx.fill();
    ctx.fillStyle=PAL.cheek;ctx.globalAlpha=.7;ctx.beginPath();ctx.arc(ex+k*hr*0.34,ey+hr*0.32,hr*0.13,0,TAU);ctx.fill();ctx.globalAlpha=1;});
  ctx.strokeStyle="#8a4a44";ctx.lineWidth=Math.max(2,s*0.05);ctx.lineCap="round";
  ctx.beginPath();ctx.arc(ex+hr*0.16,ey+hr*0.2,hr*0.22,0.15*Math.PI,0.85*Math.PI);ctx.stroke();
  ctx.restore();
}
function paint(rg,opts){const fit=fitter(cv.width,cv.height,46);ctx.imageSmoothingEnabled=true;
  drawScene(fit);drawBaby(rg,fit,opts||{});}

// ---------- engine ----------
let cur=0,raf=null,t0=null;
const stars=m=>"★".repeat(m.stars)+"☆".repeat(5-m.stars);
const faceFor=k=>k==="reach"?-1.55:0;  // rotate the face up when supine
const nearest=w=>{let b=M[0];M.forEach(m=>{if(Math.abs(m.week-w)<Math.abs(b.week-w))b=m;});return b;};
function poseAtWeek(w){const A=[{week:1,angles:NEWBORN},...M];if(w<=A[0].week)return NEWBORN;
  for(let i=0;i<A.length-1;i++)if(w>=A[i].week&&w<=A[i+1].week)return lerpAng(A[i].angles,A[i+1].angles,(w-A[i].week)/(A[i+1].week-A[i].week));
  return M[M.length-1].angles;}
function setInfo(week,title,st,starTxt,blurb){document.getElementById("wk").textContent="Week "+Math.round(week);
  document.getElementById("ttl").textContent=title;document.getElementById("stars").textContent=starTxt||"";
  const S=document.getElementById("st");S.style.display=st?"":"none";if(st){S.textContent=st.t;S.className="status "+st.c;}
  document.getElementById("cap").textContent=blurb||"";}
function advScroll(on){floorX+=on?4:0;}
function selectSkill(i){cur=i;t0=null;floorX=0;
  document.querySelectorAll("#skills .b").forEach((b,j)=>b.setAttribute("aria-current",j===i?"true":"false"));
  document.getElementById("week").value=M[i].week;label(M[i].week);loop();}
function loop(){cancelAnimationFrame(raf);t0=null;const m=M[cur];
  setInfo(m.week,m.title,{t:m.success?"Mastered":"Learning",c:m.success?"master":"learn"},stars(m),m.blurb);
  if(reduce){paint(staticRig(m.angles),{rattle:m.key==="reach",face:faceFor(m.key)});return;}
  const per=2600;(function fr(ts){if(t0===null)t0=ts;const p=((ts-t0)%per)/per;const r=rigOf(m.angles,m.key,p);
    advScroll(r.scroll);paint(r.rig,{rattle:m.key==="reach",face:faceFor(m.key)});raf=requestAnimationFrame(fr);})(0);}
function timeline(){const s=[];let prev=NEWBORN,pw=1;M.forEach(m=>{s.push({t:"trans",from:prev,to:m.angles,wf:pw,wt:m.week,dur:900,m});
  s.push({t:"demo",m,dur:2300});prev=m.angles;pw=m.week;});return s;}
function playAll(){cur=-1;cancelAnimationFrame(raf);t0=null;floorX=0;
  document.querySelectorAll("#skills .b").forEach(b=>b.setAttribute("aria-current","false"));
  const segs=timeline(),total=segs.reduce((a,x)=>a+x.dur,0);
  if(reduce){const m=M[M.length-1];paint(staticRig(m.angles),{});return;}
  (function fr(ts){if(t0===null)t0=ts;let e=(ts-t0)%total,i=0;while(e>segs[i].dur){e-=segs[i].dur;i++;}
    const sg=segs[i],p=e/sg.dur,m=sg.m;
    if(sg.t==="trans"){paint(staticRig(lerpAng(sg.from,sg.to,p)),{});const wk=lerp(sg.wf,sg.wt,p);
      setInfo(wk,"Growing up…",null,"","");document.getElementById("week").value=Math.round(wk);label(Math.round(wk));}
    else{const r=rigOf(m.angles,m.key,p);advScroll(r.scroll);paint(r.rig,{rattle:m.key==="reach",face:faceFor(m.key)});
      setInfo(m.week,m.title,{t:m.success?"Mastered":"Learning",c:m.success?"master":"learn"},stars(m),m.blurb);
      document.getElementById("week").value=m.week;label(m.week);}
    raf=requestAnimationFrame(fr);})(0);}
function scrubTo(w){cur=-1;cancelAnimationFrame(raf);t0=null;floorX=0;
  document.querySelectorAll("#skills .b").forEach(b=>b.setAttribute("aria-current","false"));
  const per=2900;(function fr(ts){if(t0===null)t0=ts;const sway=0.05*Math.sin(TAU*((ts-t0)%per)/per);
    const a=Object.assign({},poseAtWeek(w));a.torso+=sway;const ns=nearest(w),on=Math.abs(ns.week-w)<=1;
    paint(staticRig(a),{rattle:ns.key==="reach"&&on,face:on?faceFor(ns.key):0});
    setInfo(w,on?ns.title:"Developing…",on?{t:ns.success?"Mastered":"Learning",c:ns.success?"master":"learn"}:null,on?stars(ns):"",on?ns.blurb:"between milestones — the body is interpolated");
    raf=requestAnimationFrame(fr);})(0);}
function label(w){document.getElementById("wkNum").textContent=Math.round(w);document.getElementById("wkSkill").textContent=nearest(w).key;}

const sk=document.getElementById("skills");
M.forEach((m,i)=>{const b=document.createElement("button");b.className="b";b.type="button";b.textContent=m.title;
  b.addEventListener("click",()=>selectSkill(i));sk.appendChild(b);});
document.getElementById("ticks").innerHTML=M.map(m=>"<span>"+m.week+"</span>").join("");
const slider=document.getElementById("week");slider.max=M[M.length-1].week+2;
slider.addEventListener("input",e=>{label(+e.target.value);scrubTo(+e.target.value);});
document.getElementById("playAll").addEventListener("click",playAll);
selectSkill(0);
</script>
"""


def render(poses_path="babysim/poses.json", out="babysim/comic.html"):
    with open(poses_path) as fh:
        data = json.load(fh)
    for m in data["milestones"]:
        m.pop("mastered", None)
    html = HTML.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(html)} bytes, {len(data['milestones'])} skills)")
    return out


if __name__ == "__main__":
    render()
