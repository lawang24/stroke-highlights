const $ = id => document.getElementById(id);
const video = $('video');
const names = {forehand:'Forehand',backhand:'Backhand',no_shot:'No shot',unsure:'Unsure'};
const make = (tag,text) => {const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;};
let dataset, selected=0, busy=false;
const current = () => dataset.clips[selected];
const fail = message => {$('error').textContent=message||'';$('error').hidden=!message;};
async function request(url,data) {
  const response=await fetch(url,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not save. Try again.');return result;
}
function controls() {
  const done=dataset.clips.filter(c=>dataset.annotations.reviews[c.id]?.label).length;
  $('progress').textContent=`${done} / ${dataset.clips.length}`;
  $('previous').disabled=busy||selected===0;
  $('next').disabled=busy||selected===dataset.clips.length-1;
  $('approve').disabled=busy||!dataset.proposals[current().id];
  $('deny').disabled=busy;
  $('freeze').disabled=busy||done!==dataset.clips.length;
  $('download').disabled=busy;
  for(const button of $('labels').children)button.disabled=busy;
  for(const button of $('clips').children)button.disabled=busy;
}
function renderList() {
  $('clips').replaceChildren(...dataset.clips.map((clip,index)=>{
    const button=make('button');button.className=`clip${index===selected?' active':''}`;
    button.setAttribute('aria-label',`Review clip ${index+1}`);button.setAttribute('aria-current',String(index===selected));
    button.append(make('span',`Clip ${String(index+1).padStart(2,'0')}`),make('small',names[dataset.annotations.reviews[clip.id]?.label]||'Unreviewed'));
    button.onclick=()=>{if(!busy)loadClip(index);};return button;
  }));controls();
}
function loadClip(index) {
  selected=index;fail('');video.pause();video.src=`/clips/${current().id}.mp4`;video.load();
  const proposal=dataset.proposals[current().id], review=dataset.annotations.reviews[current().id];
  $('position').textContent=`CLIP ${index+1} OF ${dataset.clips.length}`;
  $('title').textContent='Watch the near-court player';
  $('duration').textContent=`${current().duration.toFixed(1)} seconds`;
  $('suggestion').textContent=proposal?names[proposal.label]:'No suggestion yet';
  $('reason').textContent=proposal?.reason||'You can still choose a label below.';
  $('correction').hidden=!!proposal;
  $('your-label').textContent=review?.label?`Your saved label: ${names[review.label]}`:'';
  $('save-status').textContent=review?.label?'Saved on this computer':'Awaiting your review';
  renderList();
}
async function decide(label,action) {
  if(busy)return;busy=true;controls();fail('');video.pause();$('save-status').textContent='Saving…';
  try {
    dataset.annotations=await request('/api/review',{clip_id:current().id,review:{label,action},revision:dataset.annotations.revision});
    $('your-label').textContent=`Your saved label: ${names[label]}`;
    $('save-status').textContent='Saved on this computer';
    // Prefer the next unfinished item; never silently approve the next suggestion.
    const next=dataset.clips.findIndex((c,i)=>i>selected&&!dataset.annotations.reviews[c.id]?.label);
    const remaining=dataset.clips.findIndex(c=>!dataset.annotations.reviews[c.id]?.label);
    if(next>=0)loadClip(next);else if(remaining>=0)loadClip(remaining);else{renderList();$('save-status').textContent='All clips reviewed. Ready to freeze.';}
  } catch(error){$('save-status').textContent='Not saved';fail(error.message);}
  finally{busy=false;controls();}
}
$('approve').onclick=()=>{const proposal=dataset.proposals[current().id];if(proposal)decide(proposal.label,'approve');};
$('deny').onclick=()=>{$('correction').hidden=false;};
for(const button of $('labels').children)button.onclick=()=>decide(button.dataset.label,'correct');
$('previous').onclick=()=>{if(!busy&&selected>0)loadClip(selected-1);};
$('next').onclick=()=>{if(!busy&&selected<dataset.clips.length-1)loadClip(selected+1);};
$('replay').onclick=()=>{video.currentTime=0;video.play().catch(e=>fail(e.message));};
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value);};
video.onloadedmetadata=()=>{video.playbackRate=Number($('speed').value);};
video.onerror=()=>fail('Could not play this clip. Check that the local review server is running.');
document.addEventListener('keydown', event=>{
  if(!dataset || busy || event.repeat || event.isComposing || event.altKey || event.ctrlKey || event.metaKey) return;
  if(event.target.closest('input, textarea, select, [contenteditable="true"]')) return;
  const key=event.key.toLowerCase();
  let button=key==='a'?$('approve'):key==='d'?$('deny'):key==='r'?$('replay'):null;
  if(!button && !$('correction').hidden && /^[1-4]$/.test(key)) button=$('labels').children[Number(key)-1];
  if(button){event.preventDefault();if(!button.disabled)button.click();}
  else if(key==='p'){
    event.preventDefault();if(video.paused)video.play().catch(e=>fail(e.message));else video.pause();
  }
});
$('download').onclick=async()=>{try{const data=await request('/api/export');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=make('a');link.href=url;link.download=`${dataset.id}-review.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){fail(error.message);}};
$('freeze').onclick=async()=>{
  if(busy)return;busy=true;controls();fail('');$('freeze-status').textContent='Checking videos and saving your labels…';
  try{const result=await request('/api/freeze',{revision:dataset.annotations.revision});$('freeze-status').textContent=`Frozen: ${result.file} · ${result.included} scored clips. No timing checks.`;}
  catch(error){$('freeze-status').textContent='';fail(error.message);}
  finally{busy=false;controls();}
};
try{dataset=await request('/api/dataset');const next=dataset.clips.findIndex(c=>!dataset.annotations.reviews[c.id]?.label);loadClip(next<0?0:next);}catch(error){fail(error.message);}
