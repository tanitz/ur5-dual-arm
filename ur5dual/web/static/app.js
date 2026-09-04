(() => {
  const $ = (q) => document.querySelector(q);
  const $$ = (q) => [...document.querySelectorAll(q)];
  // crypto.randomUUID is secure-context only, so plain http:// on a LAN address needs the fallback.
  const uuid = () => crypto.randomUUID ? crypto.randomUUID() : ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,
    c => (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));
  const clientId = sessionStorage.ur5dualClient || (sessionStorage.ur5dualClient = uuid());
  const queryToken = new URLSearchParams(location.search).get('token');
  if (queryToken) sessionStorage.ur5dualToken = queryToken;
  const accessToken = sessionStorage.ur5dualToken || '';
  if (queryToken) history.replaceState(null, '', location.pathname);
  let state = null, renderedRevision = -1, ws = null, cameraWs = null, cameraUrl = null, cameraRetry = null, held = null, heartbeat = null, editorRevision = -1, editorDirty = false;
  let commRevision = -1, commDirty = false, selectedLink = '', selectedItem = '';

  async function command(action, data={}) {
    const response = await fetch('/api/command', {method:'POST', headers:{'Content-Type':'application/json','X-UR5Dual-Token':accessToken}, body:JSON.stringify({action, client_id:clientId, ...data})});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) { const message = body.detail || `command failed (${response.status})`; $('#lastMessage').textContent = message; throw new Error(message); }
    if (body.state) render(body.state);
    return body.result;
  }
  function active(group, value, key) { $$(group).forEach(b => b.classList.toggle('active', b.dataset[key] === String(value))); }
  function option(select, value) { if (document.activeElement !== select) select.value = value ?? ''; }
  function text(id, value) { $(id).textContent = value ?? ''; }

  function render(s) {
    if (Number.isFinite(s?.revision) && s.revision < renderedRevision) return;
    if (Number.isFinite(s?.revision)) renderedRevision = s.revision;
    state = s;
    $('#linkState').textContent = 'live'; $('#linkState').classList.remove('warn');
    // Which cell this browser is driving, said in a word rather than left to
    // be guessed from two dots that a simulated arm lights up as well.
    const mode = $('#cellMode');
    mode.textContent = s.simulated ? 'SIM' : 'REAL';
    mode.classList.toggle('sim', !!s.simulated);
    mode.classList.toggle('real', !s.simulated);
    const arms = s.arms || {};
    text('#armState', ['A','B'].map(a => `${a} ${arms[a]?.connected ? '●' : '○'} ${arms[a]?.pose_text || '—'}`).join('\n'));
    text('#lastMessage', s.last_message || 'Ready');
    const p = s.program || {};
    text('#programName', p.name || 'untitled'); text('#programProblem', p.problem || '');
    $('#run').disabled = !!p.running; $('#pause').disabled = !p.running; $('#stop').disabled = !p.running;
    $('#pause').textContent = p.paused ? '▶ Resume' : '⏸ Pause';
    $('#programTable tbody').innerHTML = (p.rows || []).map((r,i) => `<tr class="${i===p.current?'current':''}"><td>${r.index}</td><td>${esc(r.kind)}</td><td>${esc(r.a)}</td><td>${esc(r.b)}</td><td>${esc(r.link)}</td></tr>`).join('');
    const files = $('#programFiles'), old = files.value;
    files.innerHTML = (p.files || []).map(n => `<option>${esc(n)}</option>`).join(''); if ([...files.options].some(o=>o.value===old)) files.value=old;
    if (!editorDirty && document.activeElement !== $('#programJson') && editorRevision !== s.revision) { $('#programJson').value = JSON.stringify(p.raw || {}, null, 2); editorRevision=s.revision; }

    const pts = s.points || [];
    text('#pointCount', `${pts.length} points`);
    $('#pointsTable tbody').innerHTML = pts.map(p => `<tr><td>${esc(p.name)}</td>${p.pose.map(v=>`<td>${Number(v).toFixed(1)}</td>`).join('')}<td><button class="danger delete-point" data-name="${attr(p.name)}">Delete</button></td></tr>`).join('');

    const links = s.comm?.links || [];
    if (!links.some(link=>link.name===selectedLink)) selectedLink=links[0]?.name||'';
    const chosen=links.find(link=>link.name===selectedLink), items=chosen?.data||[];
    if (!items.some(item=>item.name===selectedItem)) selectedItem=items[0]?.name||'';
    text('#commCount', `${links.length} machines`); text('#commStatus', s.comm?.status || '');
    $('#commTable tbody').innerHTML=links.map(link=>`<tr data-link="${attr(link.name)}" class="${link.name===selectedLink?'selected':''}"><td>${esc(link.name)}</td><td>${esc(kindName(link.kind))}</td><td>${esc(linkAddress(link))}</td></tr>`).join('');
    text('#commDataTitle', chosen?`Data on ${chosen.name}`:'Data');
    $('#commData tbody').innerHTML=items.map(item=>`<tr data-item="${attr(item.name)}" class="${item.name===selectedItem?'selected':''}"><td>${esc(item.name)}</td><td>${esc(item.direction||'both')}</td><td>${esc(itemDescription(chosen,item))}</td></tr>`).join('');
    if (!commDirty && document.activeElement!==$('#commJson') && commRevision!==s.revision) {$('#commJson').value=JSON.stringify(links,null,2);commRevision=s.revision}

    const c=s.camera||{}; active('#cameraModes button', c.mode, 'mode'); text('#cameraReading', c.reading || '—');
    option($('#cameraSource'), c.source); $('#cameraLive').textContent=c.running?'■ Stop':'▶ Live'; $('#cameraLive').className=c.running?'danger':'success';cameraStream();
    const box=$('#boxSize'), boxOld=box.value;
    box.innerHTML=`<option value="auto">Auto (depth)</option>`+(c.sizes||[]).map(v=>`<option value="${v.join(',')}">${v.join(' × ')} mm</option>`).join('');
    box.value=c.auto_size?'auto':((c.box_mm||[]).join(',')||boxOld);
    ['L','W','H'].forEach((x,i)=>{const field=$(`#box${x}`);field.disabled=!!c.auto_size;if(document.activeElement!==field)field.value=(c.box_mm||[])[i]||''});

    const j=s.jog||{}; active('#jogTargets button',j.target,'target'); active('#jogPresets button',j.preset,'preset'); option($('#jogMotion'),j.hold_mode?'hold':'step'); option($('#jogFrame'),j.frame); text('#jogSize',j.size||''); text('#jogNote',j.note||'');
    const labels=j.frame==='joint'?['J1','J2','J3','J4','J5','J6']:['X','Y','Z','RX','RY','RZ'];
    if ($('#jogGrid').dataset.labels !== labels.join(',')) { $('#jogGrid').dataset.labels=labels.join(','); $('#jogGrid').innerHTML=labels.map((label,row)=>`<div class="axis">${label}</div><button class="jog" data-row="${row}" data-sign="-1">−</button><button class="jog" data-row="${row}" data-sign="1">+</button>`).join(''); }
  }
  function kindName(kind){return {tcp:'TCP',udp:'UDP',modbus:'Modbus TCP'}[kind]||kind}
  function linkAddress(link){return `${link.host||''}:${link.port||''}${link.kind==='modbus'?`  #${link.unit??1}`:''}`}
  function itemDescription(link,item){return link?.kind==='modbus'?`${item.area||'holding'} ${item.register??0} = ${item.value??0}`:`${item.match&&item.match!=='exact'?item.match+' ':''}${JSON.stringify(item.value??'')}`}
  function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))} function attr(v){return esc(v)}
  function connect(){ const token=accessToken?`?access_token=${encodeURIComponent(accessToken)}`:''; ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws${token}`); ws.onmessage=e=>{const message=JSON.parse(e.data);if(message.type==='jog_error'){if(!message.session_id||message.session_id===held?.session_id)dropJog();text('#lastMessage',message.detail||'jog failed')}else if(message.type!=='jog_ack')render(message)}; ws.onclose=()=>{ $('#linkState').textContent='reconnecting'; $('#linkState').classList.add('warn'); dropJog(); setTimeout(connect,1000); }; }
  function cameraWanted(){return $('#camera').classList.contains('active')&&!!state?.camera?.running}
  function cameraStream(){if(cameraWanted()){if(cameraWs&&cameraWs.readyState<=WebSocket.OPEN)return;clearTimeout(cameraRetry);const token=accessToken?`?access_token=${encodeURIComponent(accessToken)}`:'';cameraWs=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/camera${token}`);cameraWs.binaryType='blob';cameraWs.onmessage=e=>{const old=cameraUrl;cameraUrl=URL.createObjectURL(e.data);$('#cameraImage').src=cameraUrl;if(old)setTimeout(()=>URL.revokeObjectURL(old),1000)};cameraWs.onclose=()=>{cameraWs=null;if(cameraWanted())cameraRetry=setTimeout(cameraStream,500)}}else{clearTimeout(cameraRetry);cameraRetry=null;if(cameraWs){cameraWs.onclose=null;cameraWs.close();cameraWs=null}if(cameraUrl){URL.revokeObjectURL(cameraUrl);cameraUrl=null}}}

  function showTab(name){$$('.side-page').forEach(x=>x.classList.toggle('active',x.id===name));active('#tabs button',name,'tab');localStorage.ur5dualWebTab=name;cameraStream()}
  $('#tabs').onclick=e=>{const b=e.target.closest('button[data-tab]');if(b)showTab(b.dataset.tab)};
  $('#swapSide').onclick=()=>{document.body.classList.toggle('side-left');localStorage.ur5dualWebSide=document.body.classList.contains('side-left')?'left':'right'};
  $('#globalStop').onclick=()=>command('stop_all'); $('#run').onclick=()=>command('program_run'); $('#pause').onclick=()=>command('program_pause'); $('#stop').onclick=()=>command('program_stop');
  $('#loadProgram').onclick=()=>command('program_load',{name:$('#programFiles').value}); $('#saveProgram').onclick=()=>command('program_save'); $('#programJson').oninput=()=>{editorDirty=true}; $('#applyProgram').onclick=()=>{try{const value=JSON.parse($('#programJson').value);command('program_set',{program:value}).then(()=>{editorDirty=false})}catch(e){text('#lastMessage','invalid program JSON')}};
  $('#points').onclick=e=>{const teach=e.target.closest('[data-teach]');if(teach){const name=prompt('Point name:');if(name)command('point_teach',{arm:teach.dataset.teach,name})}const del=e.target.closest('.delete-point');if(del&&confirm(`Delete ${del.dataset.name}?`))command('point_delete',{name:del.dataset.name})}; $('#savePoints').onclick=()=>command('points_save');
  function linksCopy(){return JSON.parse(JSON.stringify(state?.comm?.links||[]))}
  function askMachine(old={}){const link={kind:'tcp',host:'',port:2000,format:'string',encoding:'utf-8',terminator:'\\r\\n',timeout:3,unit:1,data:[],...old};const name=prompt('Machine name',link.name||'');if(name===null)return null;const kind=prompt('Protocol: tcp / udp / modbus',link.kind);if(kind===null)return null;const host=prompt('Address',link.host||'');if(host===null)return null;const port=Number(prompt('Port',kind==='modbus'?(link.port||502):(link.port||2000)));if(!['tcp','udp','modbus'].includes(kind)||!name.trim()||!host.trim()||!Number.isInteger(port))throw new Error('invalid machine fields');link.name=name.trim();link.kind=kind;link.host=host.trim();link.port=port;if(kind==='modbus'){link.unit=Number(prompt('Device id',link.unit??1))}else{link.format=prompt('Data format: string / hex',link.format||'string')||'string';link.terminator=prompt('Message ends with',link.terminator??'\\r\\n')??link.terminator}return link}
  function askItem(link,old={}){const item={direction:'both',value:link.kind==='modbus'?1:'',match:'exact',area:'holding',register:0,...old};const name=prompt('Data name',item.name||'');if(name===null)return null;const direction=prompt('Way: send / recv / both',item.direction);if(direction===null)return null;if(!name.trim()||!['send','recv','both'].includes(direction))throw new Error('invalid data fields');item.name=name.trim();item.direction=direction;if(link.kind==='modbus'){item.area=prompt('Table: coil / discrete / holding / input',item.area)||item.area;item.register=Number(prompt('Register',item.register??0));item.value=Number(prompt('Value',item.value??0))}else{item.value=prompt('Payload',item.value??'')??item.value;if(direction!=='send')item.match=prompt('Matches: exact / contains / prefix',item.match||'exact')||'exact'}return item}
  function applyLinks(links){return command('comm_set',{links})}
  $('#commTable').onclick=e=>{const row=e.target.closest('tr[data-link]');if(row){selectedLink=row.dataset.link;selectedItem='';render(state)}};$('#commData').onclick=e=>{const row=e.target.closest('tr[data-item]');if(row){selectedItem=row.dataset.item;render(state)}};
  $('#commAdd').onclick=()=>{try{const links=linksCopy(),link=askMachine();if(link){links.push(link);selectedLink=link.name;applyLinks(links)}}catch(e){text('#lastMessage',e.message)}};
  $('#commEdit').onclick=()=>{try{const links=linksCopy(),i=links.findIndex(x=>x.name===selectedLink);if(i<0)return;const edited=askMachine(links[i]);if(edited){links[i]=edited;selectedLink=edited.name;applyLinks(links)}}catch(e){text('#lastMessage',e.message)}};
  $('#commDelete').onclick=()=>{if(selectedLink&&confirm(`Delete ${selectedLink}?`))applyLinks(linksCopy().filter(x=>x.name!==selectedLink))};
  $('#dataAdd').onclick=()=>{try{const links=linksCopy(),link=links.find(x=>x.name===selectedLink);if(!link)return;const item=askItem(link);if(item){link.data=link.data||[];link.data.push(item);selectedItem=item.name;applyLinks(links)}}catch(e){text('#lastMessage',e.message)}};
  $('#dataEdit').onclick=()=>{try{const links=linksCopy(),link=links.find(x=>x.name===selectedLink),i=(link?.data||[]).findIndex(x=>x.name===selectedItem);if(i<0)return;const item=askItem(link,link.data[i]);if(item){link.data[i]=item;selectedItem=item.name;applyLinks(links)}}catch(e){text('#lastMessage',e.message)}};
  $('#dataDelete').onclick=()=>{const links=linksCopy(),link=links.find(x=>x.name===selectedLink);if(link&&selectedItem&&confirm(`Delete ${selectedItem}?`)){link.data=(link.data||[]).filter(x=>x.name!==selectedItem);applyLinks(links)}};
  $('#commTest').onclick=()=>{if(selectedLink)command('comm_test',{name:selectedLink})};$('#commSave').onclick=()=>command('comm_save');$('#commJson').oninput=()=>{commDirty=true};$('#applyComm').onclick=()=>{try{const links=JSON.parse($('#commJson').value);command('comm_set',{links}).then(()=>{commDirty=false})}catch(e){text('#lastMessage','invalid Communication JSON')}};
  $('#cameraModes').onclick=e=>{const b=e.target.closest('[data-mode]');if(b)command('camera_mode',{mode:b.dataset.mode})}; $('#cameraLive').onclick=()=>command('camera_live',{running:!state?.camera?.running}); $('#cameraSource').onchange=e=>command('camera_source',{source:e.target.value}); $('#boxSize').onchange=e=>e.target.value==='auto'?command('camera_auto_size',{enabled:true}):command('camera_box',{box_mm:e.target.value.split(',').map(Number)}); ['L','W','H'].forEach(x=>{$(`#box${x}`).onchange=()=>command('camera_box',{box_mm:['L','W','H'].map(y=>Number($(`#box${y}`).value))})});
  $('#jogTargets').onclick=e=>{const b=e.target.closest('[data-target]');if(b)command('jog_config',{target:b.dataset.target})}; $('#jogPresets').onclick=e=>{const b=e.target.closest('[data-preset]');if(b)command('jog_config',{preset:Number(b.dataset.preset)})}; $('#jogMotion').onchange=e=>command('jog_config',{hold_mode:e.target.value==='hold'}); $('#jogFrame').onchange=e=>command('jog_config',{frame:e.target.value});
  function sendJog(action,data={}){if(!ws||ws.readyState!==WebSocket.OPEN)throw new Error('realtime jog connection is not ready');ws.send(JSON.stringify({action,client_id:clientId,...data}))}
  function pressJog(button){if(held)return;held={session_id:uuid(),row:Number(button.dataset.row),sign:Number(button.dataset.sign),target:state?.jog?.target};try{sendJog('jog_press',held);heartbeat=setInterval(()=>{try{sendJog('jog_heartbeat',{session_id:held?.session_id})}catch(e){dropJog();text('#lastMessage',e.message)}},100)}catch(e){dropJog();text('#lastMessage',e.message)}}
  function dropJog(){clearInterval(heartbeat);heartbeat=null;held=null}
  function releaseJog(){if(!held)return;const old=held;dropJog();try{sendJog('jog_release',{session_id:old.session_id,target:old.target})}catch(e){text('#lastMessage',e.message)}}
  $('#jogGrid').addEventListener('pointerdown',e=>{const b=e.target.closest('.jog');if(b){e.preventDefault();b.setPointerCapture(e.pointerId);pressJog(b)}}); addEventListener('pointerup',releaseJog); addEventListener('pointercancel',releaseJog); addEventListener('blur',releaseJog); document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseJog()});
  if(localStorage.ur5dualWebSide==='left')document.body.classList.add('side-left');showTab(['points','camera','comm','jog'].includes(localStorage.ur5dualWebTab)?localStorage.ur5dualWebTab:'points');connect();
})();
