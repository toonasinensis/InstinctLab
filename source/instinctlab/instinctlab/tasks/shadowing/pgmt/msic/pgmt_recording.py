import shutil
import subprocess
import tempfile
from copy import copy
from datetime import datetime
from uuid import uuid4

import mujoco


class VideoRecorder:
    width = 960
    height = 540
    fps = 25

    def __init__(self, directory):
        self.directory = directory.resolve()
        self.process = self.renderer = None
        self.file = self.request_id = self.error = None
        self.frames = 0

    @property
    def active(self):
        return self.process is not None

    def snapshot(self):
        return dict(
            active=self.active,
            seconds=self.frames / self.fps,
            file=self.file,
            request_id=self.request_id,
            error=self.error,
        )

    def encoder_command(self, path):
        if encoder := shutil.which("ffmpeg"):
            return [
                encoder,
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
        if encoder := shutil.which("gst-launch-1.0"):
            return [
                encoder,
                "-q",
                "-e",
                "fdsrc",
                "fd=0",
                "!",
                "rawvideoparse",
                "format=rgb",
                f"width={self.width}",
                f"height={self.height}",
                f"framerate={self.fps}/1",
                "!",
                "videoconvert",
                "!",
                "video/x-raw,format=I420",
                "!",
                "x264enc",
                "speed-preset=ultrafast",
                "tune=zerolatency",
                "threads=2",
                "!",
                "mp4mux",
                "faststart=true",
                "!",
                "filesink",
                f"location={path}",
            ]
        raise RuntimeError("录制需要 FFmpeg 或带 x264enc 和 mp4mux 的 GStreamer")

    def start(self, checkpoint, terrain, level):
        if self.active:
            return
        when = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"pgmt-{checkpoint}-{terrain}-L{level}-{when}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / (name + ".mp4")
        self.temporary = self.directory / (name + ".part.mp4")
        command = self.encoder_command(self.temporary)
        self.log = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except OSError:
            self.log.close()
            raise
        self.frames = 0
        self.next_time = None

    def capture(self, clock, sim, viewer, overlay, result, show_scan, show_ghost, reference_display):
        if not self.active or (self.next_time is not None and clock + 1e-8 < self.next_time):
            return
        if self.renderer is None or self.renderer.model is not sim.model:
            self.close_renderer()
            sizes = sim.model.vis.global_
            width, height = sizes.offwidth, sizes.offheight
            sizes.offwidth, sizes.offheight = max(self.width, width), max(self.height, height)
            try:
                self.renderer = mujoco.Renderer(sim.model, height=self.height, width=self.width)
            finally:
                sizes.offwidth, sizes.offheight = width, height
        if viewer:
            with viewer.lock():
                camera, option = copy(viewer.cam), copy(viewer.opt)
        else:
            camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
            camera.azimuth, camera.elevation, camera.distance = 135, -33, 4.3
            camera.lookat[:] = sim.root_pos
            option.geomgroup[3] = 0
        self.renderer.update_scene(sim.data, camera=camera, scene_option=option)
        overlay.draw(self.renderer.scene, result, show_scan, show_ghost, reference_display, append=True)
        self.process.stdin.write(self.renderer.render().tobytes())
        self.frames += 1
        self.next_time = (clock if self.next_time is None else self.next_time) + 1 / self.fps
        if self.next_time <= clock:
            self.next_time = clock + 1 / self.fps

    def close_renderer(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def stop(self):
        if not self.active:
            return
        process, self.process = self.process, None
        try:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                status = process.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait()
                raise RuntimeError("等待 MP4 编码结束超时") from error
            if status != 0:
                self.log.seek(0)
                detail = self.log.read(2048).decode(errors="replace").strip()
                raise RuntimeError(detail or f"视频编码器退出，状态 {status}")
            if self.frames:
                self.temporary.replace(self.path)
                self.file = self.path.name
        finally:
            self.log.close()
            self.temporary.unlink(missing_ok=True)
            self.close_renderer()
