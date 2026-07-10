"""Render BabySim poses into a self-contained comic-book page (content-only HTML for the
Artifact publisher: no <html>/<head>/<body>). Inlines the poses JSON so the page is
reproducible and needs no network.

    python3 -m babysim.render          # reads babysim/poses.json -> babysim/comic.html
"""
from __future__ import annotations

import json
import os

HTML = r"""<title>BabySim — Learning to Move</title>
<style>
:root{
  --paper:#FBF3E4; --paper2:#F3E7CE; --ink:#241C3B; --muted:#6E6484;
  --coral:#FF6B5C; --sky:#4FB9C9; --sun:#FFC24B; --line:#241C3B;
  --panel:#FFFDF7; --shadow:rgba(36,28,59,.22);
  --tint-head:#FFE0D6; --tint-reach:#FFEFC7; --tint-sit:#D9F0E5;
  --tint-stand:#D8ECF6; --tint-walk:#EADFF6;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#14121F; --paper2:#1B1830; --ink:#F4EEE0; --muted:#A99FC2;
    --line:#0C0A14; --panel:#211D33; --shadow:rgba(0,0,0,.5);
    --tint-head:#3A2530; --tint-reach:#3A331E; --tint-sit:#1F3A30;
    --tint-stand:#1E3340; --tint-walk:#2C2440; }
}
:root[data-theme="light"]{ --paper:#FBF3E4; --paper2:#F3E7CE; --ink:#241C3B; --muted:#6E6484;
  --line:#241C3B; --panel:#FFFDF7; --shadow:rgba(36,28,59,.22);
  --tint-head:#FFE0D6; --tint-reach:#FFEFC7; --tint-sit:#D9F0E5; --tint-stand:#D8ECF6; --tint-walk:#EADFF6;}
:root[data-theme="dark"]{ --paper:#14121F; --paper2:#1B1830; --ink:#F4EEE0; --muted:#A99FC2;
  --line:#0C0A14; --panel:#211D33; --shadow:rgba(0,0,0,.5);
  --tint-head:#3A2530; --tint-reach:#3A331E; --tint-sit:#1F3A30; --tint-stand:#1E3340; --tint-walk:#2C2440;}

*{box-sizing:border-box}
.bs{
  --comic:"Comic Sans MS","Chalkboard SE","Comic Neue",system-ui,sans-serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-family:var(--body); color:var(--ink);
  background:
    radial-gradient(var(--paper2) 1.3px, transparent 1.4px) 0 0/14px 14px,
    var(--paper);
  min-height:100%; padding:clamp(16px,4vw,40px); line-height:1.5;
}
.bs .wrap{max-width:1120px;margin:0 auto}

/* header */
.bs header{text-align:center;margin-bottom:26px}
.bs .kicker{font-family:var(--comic);text-transform:uppercase;letter-spacing:.22em;
  font-size:.72rem;color:var(--coral);font-weight:700}
.bs h1{font-family:var(--comic);font-weight:800;line-height:.95;text-wrap:balance;
  font-size:clamp(2.6rem,8vw,4.8rem);margin:.15em 0 .1em;
  color:var(--sun);-webkit-text-stroke:2.5px var(--line);
  text-shadow:4px 5px 0 var(--shadow);letter-spacing:.01em}
.bs .lede{max-width:60ch;margin:.4em auto 0;color:var(--muted);font-size:1.02rem}
.bs .lede b{color:var(--ink)}

/* stage */
.bs .stage{position:relative;margin:22px 0 30px;border:4px solid var(--line);
  border-radius:22px;background:var(--panel);box-shadow:8px 9px 0 var(--shadow);
  overflow:hidden}
.bs .stage .bg{position:absolute;inset:0;background:
  radial-gradient(circle at 1px 1px, var(--line) 1.1px, transparent 1.4px) 0 0/12px 12px;
  opacity:.06}
.bs .stage svg{display:block;width:100%;height:min(52vh,440px)}
.bs .stagebar{display:flex;flex-wrap:wrap;align-items:center;gap:12px;
  padding:14px 18px;border-top:3px dashed var(--line);background:var(--paper2)}
.bs .stagebar .wk{font-family:var(--comic);font-weight:800;font-size:1.05rem;
  background:var(--ink);color:var(--paper);padding:3px 12px;border-radius:999px;
  font-variant-numeric:tabular-nums}
.bs .stagebar .ttl{font-family:var(--comic);font-weight:800;font-size:1.25rem;flex:1;min-width:160px}
.bs .stagebar .cap{flex-basis:100%;color:var(--muted);font-size:.95rem;margin-top:-2px}
.bs .stars{letter-spacing:2px;font-size:1.1rem;color:var(--sun);
  -webkit-text-stroke:.8px var(--line)}
.bs .status{font-family:var(--comic);font-weight:700;font-size:.75rem;text-transform:uppercase;
  letter-spacing:.08em;padding:3px 10px;border-radius:999px;border:2px solid var(--line)}
.bs .status.master{background:var(--sun);color:var(--ink)}
.bs .status.learn{background:var(--sky);color:var(--ink)}
.bs .replay{font-family:var(--comic);font-weight:700;cursor:pointer;border:3px solid var(--line);
  background:var(--coral);color:#fff;border-radius:999px;padding:6px 16px;font-size:.9rem;
  box-shadow:2px 3px 0 var(--shadow)}
.bs .replay:active{transform:translate(1px,2px);box-shadow:0 1px 0 var(--shadow)}
.bs .replay:focus-visible{outline:3px solid var(--sky);outline-offset:2px}

/* panel strip */
.bs .rail-label{font-family:var(--comic);font-weight:800;text-transform:uppercase;
  letter-spacing:.1em;font-size:.8rem;color:var(--muted);margin:6px 2px 12px}
.bs .rail{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
.bs .panel{position:relative;border:3.5px solid var(--line);border-radius:16px;
  background:var(--tint,#fff);box-shadow:5px 6px 0 var(--shadow);cursor:pointer;
  overflow:hidden;text-align:left;padding:0;color:inherit;font:inherit;
  transition:transform .12s ease}
.bs .panel:hover{transform:translate(-1px,-3px)}
.bs .panel:focus-visible{outline:3px solid var(--sky);outline-offset:3px}
.bs .panel[aria-current="true"]{outline:4px solid var(--coral);outline-offset:0}
.bs .panel .num{position:absolute;top:8px;left:10px;font-family:var(--comic);font-weight:800;
  font-size:.8rem;color:var(--ink);opacity:.5;font-variant-numeric:tabular-nums}
.bs .panel .badge{position:absolute;top:8px;right:8px;font-family:var(--comic);font-weight:800;
  font-size:.56rem;letter-spacing:.04em;background:var(--sun);color:var(--ink);
  border:2px solid var(--line);border-radius:6px;padding:2px 6px;transform:rotate(4deg)}
.bs .panel svg{display:block;width:100%;height:170px}
.bs .panel .cap{padding:10px 12px 12px;border-top:3px solid var(--line);background:var(--panel)}
.bs .panel .cap .wk{font-family:var(--comic);font-weight:800;font-size:.72rem;color:var(--coral);
  text-transform:uppercase;letter-spacing:.06em}
.bs .panel .cap .t{font-family:var(--comic);font-weight:800;font-size:1.02rem;line-height:1.1;margin:1px 0 3px}
.bs .panel .cap .s{font-size:.95rem;color:var(--sun);-webkit-text-stroke:.7px var(--line);letter-spacing:1px}

.bs .foot{margin-top:26px;text-align:center;color:var(--muted);font-size:.85rem;line-height:1.6}
.bs .foot code{font-family:ui-monospace,Menlo,Consolas,monospace;background:var(--paper2);
  padding:1px 6px;border-radius:5px;color:var(--ink)}
@media (prefers-reduced-motion:reduce){ .bs .panel{transition:none} }
</style>

<div class="bs"><div class="wrap">
  <header>
    <div class="kicker">A Reinforcement-Learning Comic</div>
    <h1>BabySim</h1>
    <p class="lede">How a mind learns to move — <b>one week at a time</b>. A tiny 2-D baby
      teaches itself each motor milestone from scratch with <b>REINFORCE</b>; the wobble you
      see is real exploration noise annealing away as the skill is mastered. Tap a week to watch.</p>
  </header>

  <div class="stage">
    <div class="bg"></div>
    <svg id="stageSvg" viewBox="0 0 800 440" role="img" aria-label="Animated baby performing the selected skill"></svg>
    <div class="stagebar">
      <span class="wk" id="stWk">WK 6</span>
      <span class="ttl" id="stTitle">Lifts head</span>
      <span class="stars" id="stStars">★★★★</span>
      <span class="status master" id="stStatus">Mastered</span>
      <button class="replay" id="replay" type="button">▶ Replay</button>
      <span class="cap" id="stCap"></span>
    </div>
  </div>

  <div class="rail-label">The first year, milestone by milestone →</div>
  <div class="rail" id="rail"></div>

  <p class="foot">
    Each panel is a real training run: a Gaussian policy over the baby's joint angles, learned by
    policy-gradient RL on a shaped reward (head height, hand-to-toy distance, balance, foot contact).
    Stars = final reward. Curriculum &amp; renderer are offline and reproducible —
    <code>python3 -m babysim.sim</code> then <code>python3 -m babysim.render</code>.
  </p>
</div></div>

<script>
const DATA = __DATA__;
const TINT = {head:"var(--tint-head)",reach:"var(--tint-reach)",sit:"var(--tint-sit)",
  stand:"var(--tint-stand)",walk:"var(--tint-walk)"};
const NS="http://www.w3.org/2000/svg";
const reduce = window.matchMedia("(prefers-reduced-motion:reduce)").matches;

// --- fit world coords (y up) into an svg viewBox (y down), including head radius + rattle
function bounds(frames, extra){
  let xs=[], ys=[];
  frames.forEach(pt=>{for(const k in pt){xs.push(pt[k][0]);ys.push(pt[k][1]);}});
  (extra||[]).forEach(p=>{xs.push(p[0]);ys.push(p[1]);});
  const HR=0.75; // head radius margin
  return {minx:Math.min(...xs)-HR,maxx:Math.max(...xs)+HR,miny:Math.min(...ys)-0.3,maxy:Math.max(...ys)+HR};
}
function fitter(b,W,H,pad){
  const sx=(W-2*pad)/(b.maxx-b.minx), sy=(H-2*pad)/(b.maxy-b.miny), s=Math.min(sx,sy);
  const ox=pad+((W-2*pad)-s*(b.maxx-b.minx))/2, oy=pad+((H-2*pad)-s*(b.maxy-b.miny))/2;
  return p=>[ox+s*(p[0]-b.minx), H-(oy+s*(p[1]-b.miny))]; // flip y
}
const lerp=(a,b,t)=>a+(b-a)*t;
function lerpPose(A,B,t){const o={};for(const k in A)o[k]=[lerp(A[k][0],B[k][0],t),lerp(A[k][1],B[k][1],t)];return o;}

function el(tag,attrs){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);return e;}
function line(svg,a,b,w,col){svg.appendChild(el("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],
  stroke:col,"stroke-width":w,"stroke-linecap":"round"}));}

// draw the baby + ground into svg for a given pose (screen coords via T), scale s (px per world unit)
function drawBaby(svg, pt, T, s, opts){
  opts=opts||{};
  const ink=getCSS("--ink"), coral=getCSS("--coral"), sky=getCSS("--sky"), sun=getCSS("--sun");
  const P=k=>T(pt[k]);
  const LW=Math.max(6,s*0.17);
  // ground line at the lowest body point
  let gy=-1e9; for(const k in pt) gy=Math.max(gy,T(pt[k])[1]);
  gy+=LW*0.5;
  svg.appendChild(el("line",{x1:20,y1:gy,x2:svg.viewBox.baseVal.width-20,y2:gy,
    stroke:ink,"stroke-width":4,"stroke-linecap":"round",opacity:.55}));
  // rattle (reach skill)
  if(opts.rattle){const r=T(opts.rattle);
    line(svg,r,[r[0]+s*0.05,r[1]+s*0.9],Math.max(4,s*0.06),ink);
    svg.appendChild(el("circle",{cx:r[0],cy:r[1],r:s*0.28,fill:sun,stroke:ink,"stroke-width":LW*0.35}));
    svg.appendChild(el("circle",{cx:r[0]-s*0.09,cy:r[1]-s*0.09,r:s*0.06,fill:"#fff",opacity:.85}));}
  // limbs (behind body): far leg/arm hint for depth
  line(svg,P("pelvis"),P("knee"),LW,ink); line(svg,P("knee"),P("foot"),LW,ink);
  line(svg,P("chest"),P("elbow"),LW,ink);  line(svg,P("elbow"),P("hand"),LW,ink);
  // onesie torso (colored)
  line(svg,P("pelvis"),P("chest"),LW*1.7,opts.color||coral);
  // booties + mitten
  svg.appendChild(el("circle",{cx:P("foot")[0],cy:P("foot")[1],r:LW*0.7,fill:sky,stroke:ink,"stroke-width":LW*0.32}));
  svg.appendChild(el("circle",{cx:P("hand")[0],cy:P("hand")[1],r:LW*0.62,fill:sky,stroke:ink,"stroke-width":LW*0.32}));
  // head
  const h=P("head"), hr=s*0.55;
  svg.appendChild(el("circle",{cx:h[0],cy:h[1],r:hr,fill:getCSS("--panel"),stroke:ink,"stroke-width":LW*0.85}));
  // hair curl
  const curl=el("path",{d:`M ${h[0]-hr*0.2} ${h[1]-hr*0.82} q ${hr*0.5} ${-hr*0.5} ${hr*0.75} ${hr*0.1}`,
    fill:"none",stroke:ink,"stroke-width":LW*0.5,"stroke-linecap":"round"}); svg.appendChild(curl);
  // face — eyes toward +x (facing forward), rosy cheeks, smile
  const ex=h[0]+hr*0.16;
  svg.appendChild(el("circle",{cx:ex-hr*0.28,cy:h[1]-hr*0.05,r:hr*0.09,fill:ink}));
  svg.appendChild(el("circle",{cx:ex+hr*0.28,cy:h[1]-hr*0.05,r:hr*0.09,fill:ink}));
  svg.appendChild(el("circle",{cx:ex-hr*0.30,cy:h[1]+hr*0.28,r:hr*0.13,fill:coral,opacity:.55}));
  svg.appendChild(el("circle",{cx:ex+hr*0.30,cy:h[1]+hr*0.28,r:hr*0.13,fill:coral,opacity:.55}));
  svg.appendChild(el("path",{d:`M ${ex-hr*0.22} ${h[1]+hr*0.18} q ${hr*0.22} ${hr*0.28} ${hr*0.44} 0`,
    fill:"none",stroke:ink,"stroke-width":LW*0.4,"stroke-linecap":"round"}));
}
function getCSS(v){return getComputedStyle(document.querySelector(".bs")).getPropertyValue(v).trim()||"#241C3B";}

function renderInto(svg, pose, skillKey, opts){
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const W=svg.viewBox.baseVal.width, H=svg.viewBox.baseVal.height;
  const extra = skillKey==="reach"?[DATA.rattle]:[];
  const b=bounds([pose],extra);
  const T=fitter(b,W,H,Math.min(W,H)*0.10);
  const s=(H-2*(Math.min(W,H)*0.10))/(b.maxy-b.miny);
  drawBaby(svg,pose,T,s,Object.assign({rattle:skillKey==="reach"?DATA.rattle:null},opts||{}));
}

// --- panel strip
const rail=document.getElementById("rail");
DATA.milestones.forEach((m,i)=>{
  const btn=document.createElement("button");
  btn.className="panel"; btn.style.setProperty("--tint",TINT[m.key]); btn.type="button";
  btn.setAttribute("aria-label",`Week ${m.week}: ${m.title}`);
  btn.innerHTML=`<span class="num">${String(i+1).padStart(2,"0")}</span>
    <span class="badge">${m.badge}</span>
    <svg viewBox="0 0 240 170" aria-hidden="true"></svg>
    <span class="cap"><span class="wk">Week ${m.week}</span>
      <span class="t">${m.title}</span><span class="s">${"★".repeat(m.stars)}${"☆".repeat(5-m.stars)}</span></span>`;
  rail.appendChild(btn);
  renderInto(btn.querySelector("svg"), m.mastered, m.key, {color:accentFor(m.key)});
  btn.addEventListener("click",()=>select(i));
});
function accentFor(k){return getCSS({head:"--coral",reach:"--sun",sit:"--sky",
  stand:"--coral",walk:"--sky"}[k]||"--coral");}  // resolve to a real hex for SVG attributes

// --- stage animation
const stageSvg=document.getElementById("stageSvg");
let cur=0, raf=null, t0=null;
function select(i){
  cur=i; const m=DATA.milestones[i];
  document.querySelectorAll(".panel").forEach((p,j)=>p.setAttribute("aria-current",j===i?"true":"false"));
  document.getElementById("stWk").textContent="WK "+m.week;
  document.getElementById("stTitle").textContent=m.title;
  document.getElementById("stStars").textContent="★".repeat(m.stars)+"☆".repeat(5-m.stars);
  const st=document.getElementById("stStatus");
  st.textContent=m.success?"Mastered":"Still learning"; st.className="status "+(m.success?"master":"learn");
  document.getElementById("stCap").textContent=m.blurb;
  play();
}
function play(){
  const m=DATA.milestones[cur];
  const seq=[...m.snaps, m.mastered, m.mastered]; // wobbly -> steady -> hold
  if(reduce){ renderInto(stageSvg,m.mastered,m.key,{color:accentFor(m.key)}); return; }
  cancelAnimationFrame(raf); t0=null;
  const seg=620; // ms per segment
  function frame(ts){
    if(t0===null)t0=ts;
    const total=(seq.length-1)*seg, el=(ts-t0)%total;
    const idx=Math.floor(el/seg), tt=(el%seg)/seg;
    const pose=lerpPose(seq[idx],seq[idx+1],tt);
    renderInto(stageSvg,pose,m.key,{color:accentFor(m.key)});
    raf=requestAnimationFrame(frame);
  }
  raf=requestAnimationFrame(frame);
}
document.getElementById("replay").addEventListener("click",play);
select(0);
</script>
"""


def render(poses_path="babysim/poses.json", out="babysim/comic.html"):
    with open(poses_path) as fh:
        data = json.load(fh)
    html = HTML.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(html)} bytes, {len(data['milestones'])} panels)")
    return out


if __name__ == "__main__":
    render()
