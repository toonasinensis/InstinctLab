import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const BLUE = "#287de1", AMBER = "#d68c21", GRAY = "#8190a3";
let manifest, dataBuffer, meshBuffer, viewport, windowIndex = 0, frame = 32, horizon = 32, playing = false;
let lastTick = 0, remainder = 0, mapRange = [-1.2, 0.2];
const data = {}, geometries = new Map();
const product = (shape) => shape.reduce((a, b) => a * b, 1);

function view(description, buffer) {
  const Type = description.dtype === "u4" ? Uint32Array : Float32Array;
  return new Type(buffer, description.offset, product(description.shape));
}
function row(name) {
  const stride = product(manifest.arrays[name].shape.slice(1));
  return data[name].subarray(windowIndex * stride, (windowIndex + 1) * stride);
}
function state(name, step) { return row(name).subarray(step * 71, (step + 1) * 71); }
function geometry(description) {
  const key = description.vertices.offset;
  if (!geometries.has(key)) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(view(description.vertices, meshBuffer), 3));
    geo.setIndex(new THREE.BufferAttribute(view(description.faces, meshBuffer), 1));
    geo.computeVertexNormals();
    geometries.set(key, geo);
  }
  return geometries.get(key);
}
function rpy(values) { return new THREE.Quaternion().setFromEuler(new THREE.Euler(...values, "ZYX")); }

class Robot {
  constructor(predicted) {
    this.group = new THREE.Group();
    this.nodes = new Map();
    this.materials = [];
    for (const link of manifest.robot.links) {
      const node = new THREE.Group();
      this.nodes.set(link.name, node);
      for (const part of link.visuals) {
        const color = new THREE.Color(...part.color.slice(0, 3));
        color.lerp(new THREE.Color(predicted ? AMBER : BLUE), predicted ? 1 : .36);
        const material = new THREE.MeshStandardMaterial({color, roughness: .54, metalness: .18});
        this.materials.push(material);
        const mesh = new THREE.Mesh(geometry(part), material);
        mesh.position.fromArray(part.position);
        mesh.quaternion.copy(rpy(part.rpy));
        mesh.scale.fromArray(part.scale);
        mesh.castShadow = !predicted;
        node.add(mesh);
      }
    }
    this.group.add(this.nodes.get(manifest.robot.root));
    this.joints = manifest.robot.joints.map((joint) => {
      const node = this.nodes.get(joint.child);
      node.position.fromArray(joint.position);
      node.quaternion.copy(rpy(joint.rpy));
      this.nodes.get(joint.parent).add(node);
      return {...joint, node, rotation: rpy(joint.rpy), axisVector: new THREE.Vector3(...joint.axis).normalize()};
    });
    this.rotation = new THREE.Quaternion();
    this.predicted = predicted;
  }
  pose(values) {
    this.group.position.fromArray(values, 0);
    this.group.quaternion.fromArray(values, 3);
    for (const joint of this.joints) {
      if (joint.index >= 0) {
        this.rotation.setFromAxisAngle(joint.axisVector, values[13 + joint.index]);
        joint.node.quaternion.copy(joint.rotation).multiply(this.rotation);
      }
    }
  }
  ghost(enabled) {
    for (const material of this.materials) {
      material.transparent = enabled;
      material.opacity = enabled ? .36 : 1;
      material.depthWrite = !enabled;
    }
  }
}

class Viewport {
  constructor() {
    this.renderer = new THREE.WebGLRenderer({canvas: $("scene"), antialias: true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor("#e9edf1");
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;
    this.renderer.setScissorTest(true);
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight("#f6f9ff", "#9aa0a8", 2.2));
    this.light = new THREE.DirectionalLight("#fff6e6", 3.0);
    this.light.castShadow = true;
    this.light.shadow.mapSize.set(2048, 2048);
    Object.assign(this.light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .1, far: 20});
    this.light.shadow.normalBias = .008;
    this.scene.add(this.light, this.light.target);
    this.world = new THREE.Group();
    this.scene.add(this.world);
    this.actual = new Robot(false);
    this.predicted = new Robot(true);
    this.world.add(this.actual.group, this.predicted.group);
    this.pathGroup = new THREE.Group();
    this.vectorGroup = new THREE.Group();
    this.world.add(this.pathGroup, this.vectorGroup);
    for (let i = 0; i < 3; i++) {
      const line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: "#bb6048", transparent: true, opacity: .8}));
      this.vectorGroup.add(line);
    }
    this.camera = new THREE.PerspectiveCamera(37, 1, .025, 100);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, $("scene"));
    this.controls.enableDamping = true;
    this.controls.minDistance = .4;
    this.controls.maxDistance = 25;
    this.mode = "overlay";
    this.focus = new THREE.Vector3();
    this.actual.group.name = "actual";
    this.predicted.group.name = "predicted";
    this.renderer.setAnimationLoop((now) => { advance(now); this.render(); });
  }
  selectWindow() {
    const w = manifest.windows[windowIndex], c = manifest.cases[w.case];
    if (this.terrain) this.world.remove(this.terrain);
    this.terrain = new THREE.Mesh(geometry(c.terrain), new THREE.MeshStandardMaterial({color: "#bdc2c5", roughness: .94, flatShading: true}));
    this.terrain.receiveShadow = true;
    this.world.add(this.terrain);
    this.world.position.set(...c.origin.map((value) => -value));
    const initial = state("actual", 0);
    this.heading = Math.atan2(2 * (initial[6] * initial[5] + initial[3] * initial[4]), 1 - 2 * (initial[4] ** 2 + initial[5] ** 2));
    this.bounds = new THREE.Box3();
    // Fit the complete valid window once. Both trajectories stay in world space.
    for (const name of ["actual", "predicted"]) {
      const values = row(name), feet = row(name + "_feet");
      for (let t = 0; t <= w.valid_steps; t++) {
        const root = new THREE.Vector3(...values.slice(t * 71, t * 71 + 3)).sub(new THREE.Vector3(...c.origin));
        this.bounds.expandByPoint(root.clone().add(new THREE.Vector3(-.65, -.65, 0)));
        this.bounds.expandByPoint(root.clone().add(new THREE.Vector3(.65, .65, .65)));
        for (let foot = 0; foot < 2; foot++) this.bounds.expandByPoint(new THREE.Vector3(...feet.slice(t * 6 + foot * 3, t * 6 + foot * 3 + 3)).sub(new THREE.Vector3(...c.origin)));
      }
    }
    this.bounds.getCenter(this.focus);
    this.light.position.copy(this.focus).add(new THREE.Vector3(3, -4, 6));
    this.light.target.position.copy(this.focus);
    this.setCamera("orbit");
    while (this.pathGroup.children.length) {
      const line = this.pathGroup.children[0];
      this.pathGroup.remove(line); line.geometry.dispose(); line.material.dispose();
    }
    for (const [name, color] of [["actual", BLUE], ["predicted", AMBER]]) {
      const states = row(name), feet = row(name + "_feet");
      for (let track = 0; track < 3; track++) {
        const positions = new Float32Array(33 * 3);
        for (let t = 0; t < 33; t++) {
          for (let axis = 0; axis < 3; axis++) positions[t * 3 + axis] = track === 0 ? states[t * 71 + axis] : feet[t * 6 + (track - 1) * 3 + axis];
          if (track) positions[t * 3 + 2] += .008;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        this.pathGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({color, transparent: true, opacity: track ? .85 : .45})));
      }
    }
  }
  setCamera(mode) {
    this.cameraMode = mode;
    const offsets = {orbit: [2.6, -1.9, 1.15], top: [.001, 0, 4.1], side: [0, -3.5, .15]};
    const direction = new THREE.Vector3(...offsets[mode]).normalize();
    if (mode !== "top") direction.applyAxisAngle(new THREE.Vector3(0, 0, 1), this.heading);
    const right = new THREE.Vector3().crossVectors(direction, this.camera.up).normalize();
    const up = new THREE.Vector3().crossVectors(right, direction).normalize();
    const aspect = $("scene").clientWidth / $("scene").clientHeight / (this.mode === "split" ? 2 : 1);
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    let distance = 2;
    for (const x of [this.bounds.min.x, this.bounds.max.x]) for (const y of [this.bounds.min.y, this.bounds.max.y]) for (const z of [this.bounds.min.z, this.bounds.max.z]) {
      const corner = new THREE.Vector3(x, y, z).sub(this.focus), depth = corner.dot(direction);
      distance = Math.max(distance, depth + Math.abs(corner.dot(right)) / (tangent * aspect), depth + Math.abs(corner.dot(up)) / tangent);
    }
    this.camera.position.copy(this.focus).addScaledVector(direction, distance * 1.08);
    this.controls.target.copy(this.focus);
    this.controls.update();
  }
  update() {
    this.actual.pose(state("actual", frame)); this.predicted.pose(state("predicted", frame));
    this.world.updateMatrixWorld(true);
    const expected = row("actual_feet"), worldPosition = new THREE.Vector3();
    const origin = manifest.cases[manifest.windows[windowIndex].case].origin;
    let fkError = 0;
    for (const [i, side] of ["left", "right"].entries()) {
      this.actual.nodes.get(`${side}_ankle_roll_link`).getWorldPosition(worldPosition);
      worldPosition.add(new THREE.Vector3(...origin));
      fkError = Math.max(fkError, worldPosition.distanceTo(new THREE.Vector3(...expected.slice(frame * 6 + i * 3, frame * 6 + i * 3 + 3))));
    }
    if (fkError > .0003) throw new Error(`Browser FK/native mismatch: ${fkError} m`);
    $("load-status").title = `Browser/native foot FK error: ${(fkError * 1000).toFixed(4)} mm`;
    this.pathGroup.visible = $("paths").checked;
    this.vectorGroup.visible = $("vectors").checked;
    for (const line of this.pathGroup.children) line.geometry.setDrawRange(0, frame + 1);
    const actual = state("actual", frame), predicted = state("predicted", frame);
    for (let i = 0; i < 3; i++) {
      const positions = i === 0 ? [...actual.slice(0, 3), ...predicted.slice(0, 3)] : [
        ...row("actual_feet").slice(frame * 6 + (i - 1) * 3, frame * 6 + i * 3),
        ...row("predicted_feet").slice(frame * 6 + (i - 1) * 3, frame * 6 + i * 3),
      ];
      this.vectorGroup.children[i].geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      this.vectorGroup.children[i].geometry.computeBoundingSphere();
    }
  }
  render() {
    const canvas = $("scene"), width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    if (this.width !== width || this.height !== height) {
      this.width = width; this.height = height; this.renderer.setSize(width, height, false);
    }
    this.controls.update();
    const split = this.mode === "split", pane = split ? Math.floor(width / 2) : width;
    this.camera.aspect = pane / height; this.camera.updateProjectionMatrix();
    this.actual.group.visible = true; this.predicted.group.visible = !split;
    this.predicted.ghost(!split);
    this.pathGroup.visible = $("paths").checked && !split;
    this.vectorGroup.visible = $("vectors").checked && !split;
    this.renderer.setViewport(0, 0, pane, height); this.renderer.setScissor(0, 0, pane, height);
    this.renderer.render(this.scene, this.camera);
    if (split) {
      this.actual.group.visible = false; this.predicted.group.visible = true;
      this.renderer.setViewport(pane, 0, width - pane, height); this.renderer.setScissor(pane, 0, width - pane, height);
      this.renderer.render(this.scene, this.camera);
    }
  }
}

function canvas(id) {
  const element = $(id), width = element.clientWidth, height = element.clientHeight;
  const dpr = Math.min(devicePixelRatio, 2);
  element.width = width * dpr; element.height = height * dpr;
  const ctx = element.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}
function plot(id, series, from = 0) {
  const {ctx, width, height} = canvas(id), left = 38, right = width - 8, top = 8, bottom = height - 18;
  const values = series.flatMap((s) => Array.from(s.values));
  let lo = Math.min(0, ...values), hi = Math.max(.001, ...values);
  const padding = (hi - lo) * .1; lo -= padding; hi += padding;
  const x = (step) => left + step / Math.max(1, cap()) * (right - left), y = (v) => bottom - (v - lo) / (hi - lo) * (bottom - top);
  ctx.fillStyle = "#f7f9fb"; ctx.fillRect(left, top, right - left, bottom - top);
  if (cap() > manifest.train_horizon) { ctx.fillStyle = "#fcf2e5"; ctx.fillRect(x(manifest.train_horizon), top, right - x(manifest.train_horizon), bottom - top); }
  ctx.font = "10px system-ui"; ctx.fillStyle = "#8b96a7";
  for (let i = 0; i <= 2; i++) {
    const value = lo + i / 2 * (hi - lo), yy = y(value);
    ctx.strokeStyle = "#e4e9ef"; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
    ctx.fillText(Math.abs(value) < .01 ? value.toFixed(3) : value.toFixed(1), 0, yy + 3);
  }
  for (const step of [0, 8, 16, 32].filter((s) => s <= cap())) ctx.fillText(String(step), x(step) - 3, height - 2);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.setLineDash(s.dashed ? [4, 3] : []); ctx.beginPath();
    s.values.forEach((value, i) => { const xx = x(i + from), yy = y(value); if (i) ctx.lineTo(xx, yy); else ctx.moveTo(xx, yy); }); ctx.stroke();
  }
  ctx.setLineDash([]); ctx.strokeStyle = "#57667b"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x(frame), top); ctx.lineTo(x(frame), bottom); ctx.stroke();
}
function heat(id, values, difference = false) {
  const element = $(id); element.width = element.height = 126;
  const ctx = element.getContext("2d");
  for (let x = 0; x < 21; x++) for (let y = 0; y < 21; y++) {
    const value = values[x * 21 + y];
    if (difference) {
      const v = Math.max(-1, Math.min(1, value / .05)), t = Math.abs(v);
      const rgb = v > 0 ? [225, 96, 70] : [49, 119, 216];
      ctx.fillStyle = `rgb(${rgb.map((c) => Math.round(247 + t * (c - 247))).join(",")})`;
    } else {
      const t = Math.max(0, Math.min(1, (value - mapRange[0]) / (mapRange[1] - mapRange[0])));
      ctx.fillStyle = `hsl(${220 - 175 * t} 66% ${42 + 21 * t}%)`;
    }
    ctx.fillRect(x * 6, (20 - y) * 6, 6, 6);
  }
}
function topPath() {
  const {ctx, width, height} = canvas("top-path"), a = row("actual"), b = row("predicted");
  const points = [a, b].flatMap((v) => Array.from({length: cap() + 1}, (_, i) => [v[i * 71], v[i * 71 + 1]]));
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(.15, Math.max(...xs) - Math.min(...xs), (Math.max(...ys) - Math.min(...ys)) * width / height) * 1.3;
  const scale = (width - 24) / span, px = (x) => width / 2 + (x - cx) * scale, py = (y) => height / 2 - (y - cy) * scale;
  ctx.fillStyle = "#f5f7fa"; ctx.fillRect(0, 0, width, height);
  for (const [values, color] of [[a, BLUE], [b, AMBER]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.beginPath();
    for (let t = 0; t <= frame; t++) { const x = px(values[t * 71]), y = py(values[t * 71 + 1]); if (t) ctx.lineTo(x, y); else ctx.moveTo(x, y); } ctx.stroke();
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(px(values[frame * 71]), py(values[frame * 71 + 1]), 3.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = "#59677c"; ctx.beginPath(); ctx.arc(px(a[0]), py(a[1]), 3, 0, Math.PI * 2); ctx.fill();
  ctx.font = "9px system-ui"; ctx.fillText(`X → / Y ↑ · ${(span * 100).toFixed(0)} cm width`, 5, height - 5);
}
function contact() {
  const {ctx, width, height} = canvas("contact");
  const actual = row("actual_contact"), predicted = data.predicted_contact ? row("predicted_contact") : null;
  const left = 36, cell = (width - left) / Math.max(1, cap());
  ctx.font = "10px system-ui"; ctx.fillStyle = "#728097"; ctx.fillText("Left", 0, 20); ctx.fillText("Right", 0, 49);
  for (let foot = 0; foot < 2; foot++) for (let t = 0; t < cap(); t++) {
    ctx.fillStyle = actual[t * 2 + foot] > .5 ? BLUE : "#e8ecf2"; ctx.fillRect(left + t * cell, 7 + foot * 29, cell - .5, 10);
    ctx.fillStyle = !predicted ? "#d7dce3" : predicted[t * 2 + foot] > .5 ? AMBER : "#e8ecf2"; ctx.fillRect(left + t * cell, 19 + foot * 29, cell - .5, 10);
  }
  ctx.strokeStyle = "#53647c"; ctx.beginPath(); ctx.moveTo(left + frame * cell, 0); ctx.lineTo(left + frame * cell, height); ctx.stroke();
}
function rewards() {
  const index = Number($("reward").value), keys = manifest.reward_keys, supported = manifest.predicted_reward_keys;
  const native = row("native_reward"), oracle = row("oracle_reward"), predicted = row("predicted_reward");
  const names = index < 0 ? supported : [keys[index]];
  const series = [[native, keys, BLUE], [oracle, keys, GRAY], [predicted, supported, AMBER]].map(([values, order, color]) => ({
    color, dashed: color === GRAY,
    values: Array.from({length: cap()}, (_, t) => names.reduce((sum, name) => sum + values[t * order.length + order.indexOf(name)], 0)),
  }));
  plot("reward-plot", series, 1);
}
function cap() { return Math.max(0, Math.min(horizon, manifest.windows[windowIndex].valid_steps)); }
function update() {
  const w = manifest.windows[windowIndex], maximum = cap(); frame = Math.max(0, Math.min(frame, maximum));
  $("scrub").max = maximum; $("scrub").value = frame;
  $("play").textContent = playing ? "Ⅱ 暂停" : "▶ 播放";
  $("root-error").textContent = `${(row("root_error")[frame] * 100).toFixed(2)} cm`;
  $("foot-error").textContent = `${(row("foot_error")[frame] * 100).toFixed(2)} cm`;
  const a = state("actual", frame), p = state("predicted", frame);
  let squared = 0; for (let i = 13; i < 42; i++) squared += (a[i] - p[i]) ** 2;
  $("joint-error").textContent = `${(Math.sqrt(squared / 29) * 180 / Math.PI).toFixed(2)}°`;
  $("step-label").textContent = `H${frame} / ${maximum}`;
  $("time").textContent = `+${(frame * manifest.dt).toFixed(2)} s · step ${w.start_step + frame}`;
  $("scene-title").textContent = `${manifest.cases[w.case].name} · 起点 step ${w.start_step} · episode ${w.episode}`;
  const terrain = manifest.cases[w.case].display_terrain;
  const layouts = {pyramid_stairs: "四面金字塔楼梯 · 中心高", inverted_pyramid_stairs: "四面倒金字塔楼梯 · 中心低", flat: "平地", rough: "粗糙地形"};
  $("terrain-description").textContent = terrain ? `${layouts[terrain.layout] || terrain.layout} · 完整 ${terrain.tile_size.join(" × ")} m 地形块${terrain.step_height ? ` · 阶高 ${(terrain.step_height * 100).toFixed(0)} cm / 阶宽 ${(terrain.step_width * 100).toFixed(0)} cm` : ""}` : "旧版局部裁剪地形";
  $("validity").textContent = maximum < horizon ? `窗口在 H${maximum} 结束：${w.terminal ? "真实 episode 终止" : "地形采样无效"}` : `有效预测 H${maximum} · ${frame <= manifest.train_horizon ? "训练范围内" : "超出 H8 训练范围"}`;
  $("window-info").textContent = `采集步 ${w.start_step}；episode 内第 ${w.episode_step} 步。终止前最后一步保留。`;
  const step = Math.max(0, frame - 1), actualMap = row("actual_map").subarray(step * 441, (step + 1) * 441), predictedMap = row("predicted_map").subarray(step * 441, (step + 1) * 441);
  if (frame > 0 && row("map_valid")[step]) {
    const difference = Float32Array.from(actualMap, (v, i) => predictedMap[i] - v);
    heat("map-actual", actualMap); heat("map-predicted", predictedMap); heat("map-difference", difference, true);
    $("map-error").textContent = `MAE ${(difference.reduce((s, v) => s + Math.abs(v), 0) / 441 * 100).toFixed(2)} cm`;
  } else {
    for (const id of ["map-actual", "map-predicted", "map-difference"]) { $(id).getContext("2d").clearRect(0, 0, $(id).width, $(id).height); }
    $("map-error").textContent = frame === 0 ? "t₀ 尚无后继" : "无效地形采样";
  }
  $("map-scale").textContent = `高程共用色标 ${mapRange.map((v) => v.toFixed(2)).join(" … ")} m；差值 −5 … +5 cm`;
  viewport.update(); topPath(); contact(); rewards();
  plot("error-plot", [
    {color: BLUE, values: Array.from(row("root_error").slice(0, maximum + 1), (v) => v * 100)},
    {color: AMBER, values: Array.from(row("foot_error").slice(0, maximum + 1), (v) => v * 100)},
  ]);
}
function selectWindow(index) {
  windowIndex = index; playing = false; frame = cap(); remainder = 0;
  const values = [...row("actual_map"), ...row("predicted_map")];
  mapRange = [Math.min(...values), Math.max(...values) + .001];
  $("window").value = index;
  viewport.selectWindow(); update();
}
function selectCase(index) {
  $("case").value = index; $("window").replaceChildren();
  for (const w of manifest.windows.filter((w) => w.case === index)) {
    const option = new Option(`step ${w.start_step} · e${w.episode} · H${w.valid_steps} · root ${w.root_end_cm.toFixed(1)} cm`, w.index);
    $("window").add(option);
  }
  const available = manifest.windows.filter((w) => w.case === index);
  const chosen = available.find((w) => w.start_step >= 32 && w.valid_steps === 32) || available[0];
  selectWindow(chosen.index);
}
function advance(now) {
  if (!manifest || !viewport) return;
  if (playing) {
    remainder += Math.min(.1, (now - lastTick) / 1000) * Number($("speed").value) / manifest.dt;
    const steps = Math.floor(remainder);
    if (steps) {
      remainder -= steps; frame += steps;
      if (frame > cap()) { if ($("loop").checked) frame = 0; else { frame = cap(); playing = false; } }
      update();
    }
  }
  lastTick = now;
}

async function main() {
  const responses = await Promise.all(["manifest.json", "data.bin", "meshes.bin"].map(async (path) => {
    const response = await fetch(path, {cache: "no-store"}); if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`); return response;
  }));
  [manifest, dataBuffer, meshBuffer] = await Promise.all([responses[0].json(), responses[1].arrayBuffer(), responses[2].arrayBuffer()]);
  if (manifest.schema !== "pgmt-wm-viewer-v1" || !manifest.windows.length) throw new Error("不支持的数据格式或无可用窗口");
  for (const [name, description] of Object.entries(manifest.arrays)) data[name] = view(description, dataBuffer);
  manifest.cases.forEach((c, i) => $("case").add(new Option(c.name, i)));
  $("reward").add(new Option(`总奖励 (${manifest.predicted_reward_keys.length} supported)`, -1));
  manifest.reward_keys.forEach((key, i) => { if (manifest.predicted_reward_keys.includes(key)) $("reward").add(new Option(key, i)); });
  $("identity").textContent = `${manifest.motion} · WM updates ${manifest.wm_updates} · ${manifest.checkpoint_sha256.slice(0, 12)}`;
  $("metadata-text").textContent = JSON.stringify({
    checkpoint: manifest.checkpoint, sha256: manifest.checkpoint_sha256, policy: manifest.policy,
    scene_seed: manifest.scene_seed, trajectory_seed: manifest.trajectory_seed,
    terrain_matches_training: manifest.terrain_sha256 === manifest.training_terrain_sha256,
    display_terrain_revision: manifest.display_terrain_revision || 1,
    display_terrain: manifest.cases.map((c) => ({case: c.name, ...c.display_terrain})),
    heightmap: manifest.map_convention, summary: manifest.summary,
    note: "离线真实采集；灰色曲线为真实状态的解析奖励，橙色为预测状态的解析奖励。高难度边缘场景使用明确标注的 +1.6m 出生偏移。未对预测施加物理修正。",
  }, null, 2);
  viewport = new Viewport();
  $("case").onchange = () => selectCase(Number($("case").value));
  $("window").onchange = () => selectWindow(Number($("window").value));
  for (const [id, direction] of [["previous", -1], ["next", 1]]) $(id).onclick = () => {
    const options = Array.from($("window").options).map((o) => Number(o.value)), index = options.indexOf(windowIndex);
    selectWindow(options[(index + direction + options.length) % options.length]);
  };
  $("worst").onclick = () => {
    const available = manifest.windows.filter((w) => w.case === Number($("case").value) && w.valid_steps >= horizon);
    if (available.length) selectWindow(available.reduce((a, b) => rowAtError(a) > rowAtError(b) ? a : b).index);
  };
  function rowAtError(w) { return data.root_error[w.index * 33 + horizon]; }
  $("play").onclick = () => { if (!cap()) return; playing = !playing; if (playing && frame === cap()) frame = 0; remainder = 0; update(); };
  for (const [id, direction] of [["back", -1], ["forward", 1]]) $(id).onclick = () => { playing = false; frame += direction; update(); };
  $("scrub").oninput = () => { playing = false; frame = Number($("scrub").value); update(); };
  for (const button of document.querySelectorAll("[data-h]")) button.onclick = () => {
    horizon = Number(button.dataset.h); playing = false; frame = cap();
    document.querySelectorAll("[data-h]").forEach((b) => b.classList.toggle("selected", b === button)); update();
  };
  for (const button of document.querySelectorAll("[data-mode]")) button.onclick = () => {
    viewport.mode = button.dataset.mode; $("split-labels").hidden = viewport.mode !== "split";
    viewport.setCamera(viewport.cameraMode);
    document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("selected", b === button));
  };
  for (const button of document.querySelectorAll("[data-camera]")) button.onclick = () => viewport.setCamera(button.dataset.camera);
  for (const id of ["paths", "vectors", "reward"]) $(id).onchange = update;
  $("details").onclick = () => $("metadata").showModal(); $("close-details").onclick = () => $("metadata").close();
  $("export-view").onclick = () => {
    viewport.render();
    const source = $("scene"), output = document.createElement("canvas");
    output.width = source.width; output.height = source.height + 110;
    const ctx = output.getContext("2d"), w = manifest.windows[windowIndex];
    ctx.fillStyle = "#f5f7fa"; ctx.fillRect(0, 0, output.width, output.height);
    ctx.drawImage(source, 0, 68);
    ctx.fillStyle = "#233044"; ctx.font = "bold 22px system-ui";
    ctx.fillText(`WM Rollout · ${manifest.motion} · ${manifest.cases[w.case].name}`, 20, 28);
    ctx.font = "16px system-ui"; ctx.fillStyle = BLUE; ctx.fillText("● Simulator", 20, 54);
    ctx.fillStyle = AMBER; ctx.fillText("● WM prediction", 160, 54);
    ctx.fillStyle = "#53647a";
    ctx.fillText(`H${frame} · root ${$("root-error").textContent} · feet ${$("foot-error").textContent} · anchor step ${w.start_step} · checkpoint ${manifest.checkpoint_sha256.slice(0, 12)}`, 20, output.height - 16);
    output.toBlob((blob) => {
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      const previous = $("export-download");
      if (previous) { URL.revokeObjectURL(previous.href); previous.remove(); }
      link.id = "export-download"; link.textContent = "保存 PNG";
      link.href = url; link.download = `wm-rollout-case${w.case}-step${w.start_step}-h${frame}.png`; link.click();
      $("export-view").after(link);
    }, "image/png");
  };
  window.addEventListener("resize", () => update());
  window.addEventListener("keydown", (event) => {
    if (["SELECT", "INPUT"].includes(event.target.tagName)) return;
    const keys = {Space: "play", ArrowLeft: "back", ArrowRight: "forward"};
    if (keys[event.code]) { event.preventDefault(); $(keys[event.code]).click(); }
  });
  $("load-status").textContent = `${manifest.windows.length} 窗口 · ${((dataBuffer.byteLength + meshBuffer.byteLength) / 1048576).toFixed(1)} MiB · 无需 GPU`;
  selectCase(1);
}
function showError(error) {
  playing = false;
  console.error(error);
  $("fatal").hidden = false;
  $("fatal").textContent = `播放器错误：${error.stack || error}\n请通过 HTTP 打开包含 vendor/ 的完整导出目录。`;
}
window.addEventListener("error", (event) => showError(event.error || event.message));
main().catch(showError);
