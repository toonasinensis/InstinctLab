/* Motion, terrain and placement controls. The native MuJoCo state stays authoritative. */
class SceneEditor {
  constructor() {
    this.catalog = null;
    this.state = null;
    this.preview = null;
    this.initialized = false;
    this.pending = null;
    this.dirty = false;
    this.revision = -1;
    this.el = id => document.getElementById(id);
    this.form = this.el('scene_form');
    this.form.addEventListener('submit', event => { event.preventDefault(); this.apply(false); });
    this.el('apply_play').onclick = () => this.apply(false);
    this.el('apply_pause').onclick = () => this.apply(true);
    this.el('map_center').onclick = () => this.setPosition(0, 0);
    this.el('motion_select').onchange = () => {
      this.el('motion_start').value = '0.1';
      this.updateMotion(); this.edited();
    };
    for (const id of ['terrain_select', 'level_select']) this.el(id).onchange = () => {
      this.syncTerrainLevel();
      this.requestPreview(); this.updateMotion(); this.edited();
    };
    this.el('motion_start').oninput = () => { this.updateMotion(); this.edited(); };
    this.el('motion_slider').oninput = () => {
      this.el('motion_start').value = this.el('motion_slider').value;
      this.updateMotion(); this.edited();
    };
    for (const id of ['spawn_x','spawn_y','spawn_heading']) this.el(id).oninput = () => {
      this.el('heading_slider').value = this.el('spawn_heading').value;
      this.edited(); this.drawMap();
    };
    this.el('heading_slider').oninput = () => {
      this.el('spawn_heading').value = this.el('heading_slider').value;
      this.edited(); this.drawMap();
    };
    const canvas = this.el('terrain_map');
    const select = event => {
      if (this.pending || !this.preview) return;
      const box = canvas.getBoundingClientRect();
      const x = (event.clientX-box.left)*canvas.width/box.width;
      const y = (event.clientY-box.top)*canvas.height/box.height;
      if (x < 50 || x > 450 || y < 12 || y > 412) return;
      this.setPosition(Math.max(-3.9, Math.min(3.9, (x-50)/400*8.1-4.05)),
                       Math.max(-3.9, Math.min(3.9, 4.05-(y-12)/400*8.1)));
    };
    canvas.onpointerdown = event => { canvas.setPointerCapture(event.pointerId); select(event); };
    canvas.onpointermove = event => { if (event.buttons === 1) select(event); };
    this.loadCatalog();
  }

  feedback(text, error=false) {
    this.el('scene_feedback').textContent = text;
    this.el('scene_feedback').style.color = error ? '#ff8b8b' : '#67dabc';
  }

  async loadCatalog() {
    try {
      const response = await fetch('/catalog');
      if (!response.ok) throw new Error('动作列表读取失败，请刷新页面');
      this.catalog = await response.json();
      const select = this.el('motion_select');
      const groups = new Map();
      for (const motion of this.catalog.motions) {
        if (!groups.has(motion.category)) {
          const group = document.createElement('optgroup'); group.label = motion.category;
          groups.set(motion.category, group); select.appendChild(group);
        }
        const option = document.createElement('option'); option.value = motion.name;
        option.textContent = motion.name.replace('.named.npz','').replace('.npz','');
        groups.get(motion.category).appendChild(option);
      }
      for (const terrain of this.catalog.terrains) {
        const option = document.createElement('option'); option.value = terrain.value;
        option.textContent = terrain.label; this.el('terrain_select').appendChild(option);
      }
      for (const level of this.catalog.levels) {
        const option = document.createElement('option'); option.value = level; option.textContent = 'L'+level;
        this.el('level_select').appendChild(option);
      }
      this.el('catalog_count').textContent = `${this.catalog.motions.length} 个动作 · ${this.catalog.terrains.length} 种地形`;
      if (this.state) this.update(this.state);
    } catch (error) { this.feedback(error.message, true); }
  }

  selectedMotion() { return this.catalog?.motions.find(m => m.name === this.el('motion_select').value); }

  syncTerrainLevel() {
    const kind = this.el('terrain_select').value;
    const fixed = this.catalog?.terrains.find(t => t.value === kind)?.fixed_level;
    if (fixed !== null && fixed !== undefined) this.el('level_select').value = String(fixed);
    this.el('level_select').disabled = !!this.pending || (fixed !== null && fixed !== undefined);
    this.el('level_select').parentElement.hidden = kind === 'box_test';
  }

  updateMotion() {
    const motion = this.selectedMotion();
    if (!motion) return;
    const maximum = motion.max_frame/motion.fps;
    this.el('motion_start').max = maximum;
    this.el('motion_slider').max = maximum;
    this.el('motion_slider').step = 1/motion.fps;
    this.el('motion_slider').value = this.el('motion_start').value;
    const frame = Math.round(Number(this.el('motion_start').value)*motion.fps);
    this.el('motion_duration').textContent = `${motion.seconds.toFixed(1)} s / 第 ${frame} 帧`;
    this.el('ground_notice').textContent = motion.ground && this.el('terrain_select').value !== 'flat'
      ? '这个地面动作在训练中只配平地或最低级粗糙地形；当前组合用于观察地形外推表现。' : '';
  }

  edited() {
    this.dirty = true;
    if (!this.pending) this.feedback('有未应用的设置；点击“放置并暂停”或“应用并播放”。');
  }

  setPosition(x, y) {
    this.el('spawn_x').value = Number(x.toFixed(3));
    this.el('spawn_y').value = Number(y.toFixed(3));
    this.edited(); this.drawMap();
  }

  async requestPreview() {
    const terrain = this.el('terrain_select').value, level = Number(this.el('level_select').value);
    if (!terrain) return;
    const key = terrain + ':' + level;
    this.previewKey = key;
    try {
      const response = await fetch(`/preview?terrain=${encodeURIComponent(terrain)}&level=${level}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '地形预览失败');
      if (this.previewKey !== key) return;
      this.preview = data;
      const detail = data.detail;
      this.el('terrain_detail').textContent = detail.box_heights_m
        ? '固定 10 / 20 / 30 / 40 cm · 每箱 1×1 m · 2×2 排列，中心间距 2.5 m'
        : detail.step_height_m !== undefined
        ? `每级台阶 ${(detail.step_height_m*100).toFixed(0)} cm · 宽 ${(detail.step_width_m*100).toFixed(0)} cm`
        : detail.grid_height_range_m ? `方块宽 45 cm · 地面高度 ±${(detail.grid_height_range_m[1]*100).toFixed(1)} cm`
        : '平地 · 高度 0 m';
      this.drawMap();
    } catch (error) { this.feedback(error.message, true); }
  }

  setBusy(busy) {
    for (const element of this.form.querySelectorAll('input,select,button')) element.disabled = busy;
    if (!busy) this.syncTerrainLevel();
  }

  async apply(paused) {
    if (!this.initialized || this.pending || !this.form.reportValidity()) return;
    const motion = this.selectedMotion();
    const settings = {
      motion: motion.name, terrain: this.el('terrain_select').value,
      level: Number(this.el('level_select').value), start_frame: Math.round(Number(this.el('motion_start').value)*motion.fps),
      x: Number(this.el('spawn_x').value), y: Number(this.el('spawn_y').value),
      heading_deg: Number(this.el('spawn_heading').value), paused
    };
    this.setBusy(true); this.feedback('正在加载动作并按地面高度放置机器人…');
    this.pending = 'sending';
    try {
      const response = await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({command:'scene', settings})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '设置没有应用');
      this.pending = result.request_id;
    } catch (error) {
      this.pending = null; this.setBusy(false); this.feedback(error.message, true);
    }
  }

  populate(scene) {
    this.el('motion_select').value = scene.motion;
    this.el('terrain_select').value = scene.terrain;
    this.el('level_select').value = scene.level;
    this.syncTerrainLevel();
    this.el('motion_start').value = scene.start_frame/scene.fps;
    for (const [id,value] of [['spawn_x',scene.x],['spawn_y',scene.y],['spawn_heading',scene.heading_deg]])
      this.el(id).value = Number(value.toFixed(3));
    this.el('heading_slider').value = this.el('spawn_heading').value;
    this.updateMotion(); this.requestPreview();
    this.setBusy(false); this.dirty = false; this.initialized = true;
  }

  update(state) {
    this.state = state;
    if (!this.catalog || !state.scene) return;
    const result = state.control_status;
    const acknowledged = this.pending && result?.request_id === this.pending;
    if (acknowledged) {
      this.pending = null; this.setBusy(false);
      this.feedback(result.message, result.status === 'error');
    }
    if (!this.initialized || (state.scene.revision !== this.revision && (!this.dirty || acknowledged))) {
      this.populate(state.scene);
      this.revision = state.scene.revision;
    }
    const s = state.scene;
    const label = this.catalog.terrains.find(t => t.value === s.terrain)?.label || s.terrain;
    const difficulty = s.terrain === 'box_test' ? '固定布局' : `L${s.level}`;
    this.el('scene_current').textContent = `当前播放：${s.motion} · ${label} ${difficulty} · 初态 (${s.x.toFixed(2)}, ${s.y.toFixed(2)}) m / ${s.heading_deg.toFixed(1)}° · 放置时地面 z=${s.ground_z_m.toFixed(3)} m，骨盆 z=${s.robot_z_m.toFixed(3)} m`;
    this.drawMap();
  }

  heightAt(x,y) {
    let height = this.preview.plane_height;
    for (const box of this.preview.boxes)
      if (Math.abs(x-box.x) <= box.sx/2 && Math.abs(y-box.y) <= box.sy/2) height = Math.max(height,box.top);
    return height;
  }

  drawMap() {
    if (!this.preview) return;
    const canvas = this.el('terrain_map'), ctx = canvas.getContext('2d'), p = this.preview;
    const ox=50, oy=12, size=400, cell=size/81;
    const px = x => ox+(x+4.05)/8.1*size, py = y => oy+(4.05-y)/8.1*size;
    const values = p.heights.flat(), lo=Math.min(...values), hi=Math.max(...values);
    ctx.clearRect(0,0,480,450);
    for (let row=0;row<81;row++) for (let col=0;col<81;col++) {
      const t=(p.heights[row][col]-lo)/Math.max(.01,hi-lo);
      ctx.fillStyle=`rgb(${32+90*t},${60+105*t},${83+78*t})`;
      ctx.fillRect(ox+col*cell,oy+(80-row)*cell,cell+.25,cell+.25);
    }
    ctx.strokeStyle='#90b4cf44'; ctx.lineWidth=1;
    for (const tick of [-4,-2,0,2,4]) {
      ctx.beginPath();ctx.moveTo(px(tick),oy);ctx.lineTo(px(tick),oy+size);ctx.stroke();
      ctx.beginPath();ctx.moveTo(ox,py(tick));ctx.lineTo(ox+size,py(tick));ctx.stroke();
    }
    ctx.fillStyle='#b6c9d9';ctx.font='12px sans-serif';
    for (const v of [-4,0,4]) {ctx.fillText(v,px(v)-4,434);ctx.fillText(v,25,py(v)+4);}
    ctx.fillText('X / m →',386,450);ctx.fillText('Y / m',0,12);
    ctx.save();
    for (const box of p.detail.test_boxes || []) {
      const [sx,sy] = p.detail.box_footprint_m, rgba = box.rgba;
      ctx.fillStyle = `rgb(${rgba.slice(0,3).map(v=>Math.round(v*255)).join(',')})`;
      ctx.fillRect(px(box.x-sx/2),py(box.y+sy/2),sx/8.1*size,sy/8.1*size);
      ctx.fillStyle = '#08121cdd';ctx.fillRect(px(box.x)-23,py(box.y)-9,46,18);
      ctx.fillStyle = '#fff';ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(box.label,px(box.x),py(box.y));
    }
    ctx.restore();
    const arrow = (x,y,heading,color) => {
      if (![x,y,heading].every(Number.isFinite)) return;
      const a=heading*Math.PI/180, sx=px(x),sy=py(y), ex=sx+27*Math.cos(a),ey=sy-27*Math.sin(a);
      ctx.strokeStyle='#08121c';ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex,ey);ctx.stroke();
      ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(ex,ey);ctx.stroke();
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(sx,sy,5,0,2*Math.PI);ctx.fill();
      ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-10*Math.cos(a-.4),ey+10*Math.sin(a-.4));
      ctx.lineTo(ex-10*Math.cos(a+.4),ey+10*Math.sin(a+.4));ctx.closePath();ctx.fill();
    };
    const same = this.state?.terrain === p.terrain && this.state?.level === p.level;
    if (same) arrow(this.state.root_pos[0],this.state.root_pos[1],this.state.root_heading_deg,'#64e1bf');
    const x=Number(this.el('spawn_x').value), y=Number(this.el('spawn_y').value);
    arrow(x,y,Number(this.el('spawn_heading').value),'#ffc45d');
    const z=this.heightAt(x,y);
    const terrainText = p.terrain === 'box_test' ? '箱台测试（固定布局）' : `${p.terrain} L${p.level}`;
    this.el('map_height').textContent=`待应用地形：${terrainText} · 放置点 (${x.toFixed(2)}, ${y.toFixed(2)}) m · 地面 z=${z.toFixed(3)} m。点击或拖动黄点选位置。${same?'':' 当前机器人位于另一地形，绿点暂不显示。'}`;
  }
}
window.sceneEditor = new SceneEditor();

class ResetTimer {
  constructor() {
    this.input = document.getElementById('reset_seconds');
    this.button = document.getElementById('apply_reset_time');
    this.feedback = document.getElementById('reset_feedback');
    this.pending = null;
    this.dirty = false;
    this.input.oninput = () => { this.dirty = true; this.feedback.textContent = '尚未应用'; };
    document.getElementById('reset_form').onsubmit = event => { event.preventDefault(); this.apply(); };
  }

  async apply() {
    if (this.pending || !this.input.reportValidity()) return;
    this.pending = 'sending';
    this.input.disabled = this.button.disabled = true;
    this.feedback.textContent = '正在更新…';
    try {
      const response = await fetch('/control', {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({command:'reset_time', seconds:Number(this.input.value)})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'RESET 时间设置失败');
      this.pending = result.request_id;
    } catch (error) {
      this.pending = null;
      this.input.disabled = this.button.disabled = false;
      this.feedback.textContent = error.message;
    }
  }

  update(state) {
    if (state.reset_seconds === undefined) return;
    const status = state.timer_status;
    if (this.pending && status?.request_id === this.pending) {
      this.pending = null; this.dirty = false;
      this.input.disabled = this.button.disabled = false;
      this.feedback.textContent = status.message;
    }
    if (!this.dirty && !this.pending) this.input.value = state.reset_seconds;
    const elapsed = state.metrics.episode_time_s;
    document.getElementById('reset_clock').textContent = state.reset_seconds > 0
      ? `当前 ${elapsed.toFixed(1)} / ${state.reset_seconds} s · 剩余 ${Math.max(0,state.reset_seconds-elapsed).toFixed(1)} s`
      : `定时重置已关闭 · 本轮 ${elapsed.toFixed(1)} s`;
    const labels = {time_out:'达到设定时间',motion_end:'参考动作播放结束',tile_exit:'离开地形块',
      root_height:'根部高度误差过大',root_tilt:'身体倾斜误差过大',body_error:'身体跟踪误差过大'};
    const reset = state.last_reset;
    document.getElementById('last_reset').textContent = reset
      ? `最近自动 RESET：${reset.reasons.map(r=>labels[r] || r).join('、')} · 本轮运行 ${reset.episode_time_s.toFixed(2)} s（episode ${reset.episode}）`
      : '本次场景尚未自动 RESET。';
  }
}
window.resetTimer = new ResetTimer();

class ReferenceDisplay {
  constructor() {
    this.select = document.getElementById('reference_display');
    this.feedback = document.getElementById('reference_feedback');
    this.pending = null;
    this.select.onchange = () => this.apply();
  }

  async apply() {
    if (this.pending) return;
    this.pending = 'sending';
    this.select.disabled = true;
    this.feedback.textContent = '正在切换显示…';
    try {
      const response = await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({command:'reference_display', mode:this.select.value})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '参考显示切换失败');
      this.pending = result.request_id;
    } catch (error) {
      this.pending = null;
      this.select.disabled = false;
      this.feedback.textContent = error.message;
    }
  }

  update(state) {
    if (!state.reference_display) return;
    const status = state.reference_display_status;
    if (this.pending && status?.request_id === this.pending) {
      this.pending = null;
      this.select.disabled = false;
      this.feedback.textContent = status.message;
    }
    if (!this.pending) {
      this.select.value = state.reference_display;
      const signature = JSON.stringify(status);
      if (signature !== this.lastStatus) {
        this.feedback.textContent = status?.message || '';
        this.lastStatus = signature;
      }
    }
    document.getElementById('ghost').classList.toggle('active', state.show_ghost);
    document.getElementById('ghost').setAttribute('aria-pressed', String(state.show_ghost));
    const ghost = state.reference_ghost;
    const explanation = state.reference_display === 'source'
      ? '原始 NPZ：保留文件中的 Z 高度，不加地形支撑高度；在箱子或楼梯处可能穿入地形。'
      : '地形适配：NPZ 的 Z 高度加上当前地形支撑高度。';
    document.getElementById('reference_help').textContent = `${state.show_ghost ? '' : '参考机器人已隐藏，可按 G 显示。'}${explanation} 两种显示均与当前机器人 XY / yaw 对齐；仅改变青色参考，不重置、不改变策略输入。当前参考骨盆 Z=${ghost.pelvis_z_m.toFixed(3)} m，NPZ Z=${ghost.npz_pelvis_z_m.toFixed(3)} m，地形支撑 Z=${ghost.support_z_m.toFixed(3)} m。`;
  }
}
window.referenceDisplay = new ReferenceDisplay();
