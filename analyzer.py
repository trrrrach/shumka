import os
import sys
import math
import shutil
import argparse
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta
from statistics import median
from zipfile import ZipFile, ZIP_DEFLATED
from pydub import AudioSegment

# ========= опционально для датчиков =========
try:
    import psutil  # pip install psutil
except Exception:
    psutil = None

# ===================== НАСТРОЙКИ ПО УМОЛЧАНИЮ =====================

# Источник: папка с видео ИЛИ .zip/.rar архив
VIDEO_SOURCE = r"C:\VIDEO"    # можно переопределить аргументом командной строки

# Путь к 7-Zip (для .rar). Можно переопределить аргументом --seven_zip.
SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"

# Какие расширения считаем видео
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# Режим вывода результатов
#  - "new_root": каждый запуск в новую папку analyzed_YYYYMMDD_HHMMSS
#  - "purge":    пишем в папку analyzed, подпапка каждого видео предварительно очищается
BASE_OUTPUT  = "analyzed"
OUTPUT_MODE  = "new_root"

# Что сохранять дополнительно
SAVE_ORIGINAL_VIDEO = True          # класть копию исходника в подпапку результата
ORIGINAL_AUDIO_SAVE = "mp3"         # "wav" | "mp3" | "both"
ORIGINAL_AUDIO_MP3_BITRATE = "96k"  # для mp3

# Архивация результатов (экономия места)
ARCHIVE_RESULTS = True              # упаковать подпапку каждого видео в ZIP
DELETE_RESULT_DIR_AFTER_ZIP = True  # удалить подпапку после упаковки (оставить только ZIP)

# Режим нарезки видео:
#  - "fast_copy": без перекодирования (очень быстро), выход .avi, границы по ключкадрам
#  - "h264":      перекодирование в .mp4 (точнее/совместимее), медленнее
SEGMENT_MODE  = "fast_copy"
FAST_MODE     = True       # быстрый seek (-ss до -i): быстрее, но границы менее точные

# Параметры перекодирования (если SEGMENT_MODE="h264")
VID_PRESET    = "superfast"  # ultrafast/superfast/veryfast/...
CRF_VIDEO     = "30"         # 23..35 (ниже=лучше/больше)
AUDIO_BITRATE = "96k"

# Детектор звука
WIN_MS            = 25     # окно RMS (мс)
HOP_MS            = 25     # шаг окна (мс)
CALIBRATION_SEC   = 10     # 0=берём весь файл; >0=первые N сек для baseline
BASELINE_METHOD   = "p20"  # "median" или "p20" (если CALIBRATION_SEC==0)
THRESH_DELTA_DB   = 3.0    # порог: baseline + Δ (дБ)
HYST_DB           = 1.5    # гистерезис: закрытие события при (baseline+Δ−HYST)
MIN_EVENT_MS      = 300    # мин. длительность события (мс)

# >>> «сделать подлиннее» вырезки:
KEEP_SILENCE_MS       = 2000   # было 1000 — теперь по 2с поля вокруг события
EXTRA_EVENT_EXPAND_MS = 500    # дополнительно удлиним каждый кусок на 0.5с с КАЖДОЙ стороны
MERGE_GAP_MS          = 300    # склейка событий, если пауза <= этого (мс)

# Если ничего не найдено — сохраняем самый громкий кусок такой длины:
FALLBACK_SEC      = 2.0

# ========== ПРОГРЕСС И СИСТЕМНЫЕ ДАТЧИКИ ==========
SHOW_PROGRESS         = True
SHOW_SYSTEM_STATS     = True       # требует psutil (иначе отключится само)
STATS_INTERVAL_SEC    = 5
SHOW_CPU_PERCENT      = True
SHOW_RAM              = True
SHOW_DISK_IO          = True
SHOW_TEMPERATURES     = True       # на Windows часто пусто

# ===================== ЛОГГИРОВАНИЕ =====================
# лог на консоль и в файл (в папке результатов: analyzer.log)
import logging
LOGGER = logging.getLogger("analyzer")
LOGGER.setLevel(logging.INFO)

def add_console_handler():
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    LOGGER.addHandler(ch)

def add_file_handler(log_path):
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(fh)

def log(msg): LOGGER.info(msg)
def logw(msg): LOGGER.warning(msg)
def loge(msg): LOGGER.error(msg)

# ===================== УТИЛИТЫ =====================

def build_output_root(base_output: str, mode: str) -> str:
    if mode.lower() == "new_root":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = f"{base_output}_{stamp}"
        os.makedirs(root, exist_ok=True)
        return root
    else:
        os.makedirs(base_output, exist_ok=True)
        return base_output

def ensure_clean_video_dir(dir_path: str, mode: str):
    if mode.lower() == "purge":
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def ms_to_hhmmss_ms(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def fmt_td(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    td = timedelta(seconds=int(seconds))
    return str(td)

def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)

def short_time_rms_dbfs(audio: AudioSegment, win_ms=50, hop_ms=50):
    """Возвращает (times_sec, rms_db). times — центр окна в секундах."""
    n = max(1, math.ceil((len(audio) - win_ms) / hop_ms) + 1)
    times, rms_db = [], []
    for i in range(n):
        start = i * hop_ms
        end = min(len(audio), start + win_ms)
        seg = audio[start:end]
        rms = seg.rms or 1
        dbfs = 20 * math.log10(rms / (seg.max_possible_amplitude or 1))
        times.append((start + end) / 2000.0)
        rms_db.append(dbfs)
    return times, rms_db

def detect_exceedances(rms_db, hop_ms, baseline_db, delta_db, hyst_db, min_event_ms):
    up_th = baseline_db + delta_db
    down_th = up_th - hyst_db
    events = []
    in_evt = False
    evt_start_idx = None
    for i, val in enumerate(rms_db):
        if not in_evt and val > up_th:
            in_evt = True
            evt_start_idx = i
        elif in_evt and val < down_th:
            start_ms = evt_start_idx * hop_ms
            end_ms = i * hop_ms
            if (end_ms - start_ms) >= min_event_ms:
                events.append([start_ms, end_ms])
            in_evt = False
            evt_start_idx = None
    if in_evt and evt_start_idx is not None:
        start_ms = evt_start_idx * hop_ms
        end_ms = len(rms_db) * hop_ms
        if (end_ms - start_ms) >= min_event_ms:
            events.append([start_ms, end_ms])
    return events

def pad_and_merge(chunks, pad_ms, merge_gap_ms=0):
    if not chunks:
        return []
    padded = [[max(0, s - pad_ms), e + pad_ms] for s, e in chunks]
    padded.sort()
    merged = [padded[0]]
    for s, e in padded[1:]:
        ls, le = merged[-1]
        if s <= le + merge_gap_ms:
            merged[-1][1] = max(le, e)
        else:
            merged.append([s, e])
    return merged

def expand_chunks(chunks, expand_ms, total_ms):
    """Добавляет по expand_ms с каждой стороны (после склейки)."""
    if not chunks or expand_ms <= 0:
        return chunks
    out = []
    for s, e in chunks:
        s2 = max(0, s - expand_ms)
        e2 = min(total_ms, e + expand_ms)
        out.append([s2, e2])
    return out

# ===================== FFmpeg =====================

def run_ffmpeg(cmd, log_prefix="ffmpeg"):
    """Запускает FFmpeg, логирует команду и вывод. Поднимает исключение при ошибке."""
    log(f"[{log_prefix}] cmd: " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        log(f"[{log_prefix}] stdout:\n{res.stdout.strip()}")
    if res.stderr:
        log(f"[{log_prefix}] stderr:\n{res.stderr.strip()}")
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {res.returncode}")

def ffmpeg_extract_wav_mono(input_path: str, wav_path: str):
    """Быстро извлекаем полное аудио в WAV (моно 16 кГц)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path
    ]
    run_ffmpeg(cmd, "extract_wav")

def ffmpeg_save_mp3_from_wav(wav_path: str, mp3_path: str, bitrate="96k"):
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", wav_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", bitrate,
        mp3_path
    ]
    run_ffmpeg(cmd, "wav_to_mp3")

def ffmpeg_cut_h264(input_path: str, output_path: str, start_s: float, end_s: float, fast: bool):
    common = [
        "-vf", "format=yuv420p,setsar=1",
        "-c:v", "libx264",
        "-preset", VID_PRESET,
        "-crf", CRF_VIDEO,
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        "-threads", "0",
        output_path
    ]
    if fast:
        cmd = ["ffmpeg","-hide_banner","-y","-ss",f"{start_s}","-to",f"{end_s}","-i",input_path,*common]
    else:
        cmd = ["ffmpeg","-hide_banner","-y","-i",input_path,"-ss",f"{start_s}","-to",f"{end_s}",*common]
    run_ffmpeg(cmd, "cut_h264")

def ffmpeg_cut_copy(input_path: str, output_path: str, start_s: float, end_s: float):
    """Нарезка без перекодирования (очень быстро). Выход .avi. Границы — по ключкадрам."""
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{start_s}",
        "-to", f"{end_s}",
        "-i", input_path,
        "-c", "copy",
        "-map", "0:v:0?", "-map", "0:a:0?",
        "-avoid_negative_ts", "1",
        output_path
    ]
    run_ffmpeg(cmd, "cut_copy")

# ===================== Архивирование =====================

def zip_dir(src_dir: str, zip_path: str):
    """Запаковать содержимое папки src_dir в zip_path."""
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, start=src_dir)
                zf.write(full, arcname=rel)

# ===================== Источники: папка / zip / rar =====================

def collect_worklist_from_folder(folder_path):
    items = []
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(VIDEO_EXTS):
            items.append(("folder", os.path.join(folder_path, name), name))
    return items

def collect_worklist_from_zip(zip_path):
    items = []
    with ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(VIDEO_EXTS):
                items.append(("zip", info.filename, info.filename))
    items.sort(key=lambda x: x[1])
    return items

def list_rar_entries(seven_zip_exe: str, rar_path: str):
    """Возвращает список путей внутри RAR, используя 7z l -slt (строки 'Path = ...')."""
    result = subprocess.run([seven_zip_exe, "l", "-slt", rar_path],
                            capture_output=True, text=True, check=True)
    paths = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            p = line[len("Path = "):].strip()
            if p and not p.endswith("\\") and not p.endswith("/"):
                paths.append(p)
    return paths

def collect_worklist_from_rar(rar_path, seven_zip_exe):
    entries = list_rar_entries(seven_zip_exe, rar_path)
    entries = [e for e in entries if e.lower().endswith(VIDEO_EXTS)]
    entries.sort()
    return [("rar", e, e) for e in entries]

def extract_one_from_zip(zip_path, arcname):
    tmp_dir = tempfile.mkdtemp(prefix="vid_extract_")
    out_path = os.path.join(tmp_dir, os.path.basename(arcname))
    with ZipFile(zip_path, "r") as zf:
        with zf.open(arcname) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path, tmp_dir

def extract_one_from_rar(rar_path, arcname, seven_zip_exe):
    tmp_dir = tempfile.mkdtemp(prefix="vid_extract_")
    subprocess.run([seven_zip_exe, "e", "-y", rar_path, arcname, f"-o{tmp_dir}"], check=True)
    out_path = os.path.join(tmp_dir, os.path.basename(arcname))
    return out_path, tmp_dir

# ===================== Системные датчики (поток) =====================

def sensors_thread(stop_event: threading.Event):
    if not psutil:
        log("[i] psutil не установлен — монитор ресурсов отключён. Установите: pip install psutil")
        return
    last_disk = psutil.disk_io_counters() if SHOW_DISK_IO else None
    last_time = time.time()
    while not stop_event.wait(STATS_INTERVAL_SEC):
        try:
            line = []
            if SHOW_CPU_PERCENT:
                cpu = psutil.cpu_percent(interval=None)
                line.append(f"CPU: {cpu:4.1f}%")
            if SHOW_RAM:
                vm = psutil.virtual_memory()
                line.append(f"RAM: {vm.used//(1024**2)} / {vm.total//(1024**2)} MB ({vm.percent:.0f}%)")
            if SHOW_DISK_IO:
                now = time.time()
                cur = psutil.disk_io_counters()
                dt = max(1e-6, now - last_time)
                r_mb_s = (cur.read_bytes - last_disk.read_bytes) / dt / (1024**2)
                w_mb_s = (cur.write_bytes - last_disk.write_bytes) / dt / (1024**2)
                line.append(f"DiskIO: R {r_mb_s:5.1f} MB/s | W {w_mb_s:5.1f} MB/s")
                last_disk, last_time = cur, now
            if SHOW_TEMPERATURES and hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures(fahrenheit=False)
                flat = []
                for k, arr in temps.items():
                    if not arr: continue
                    tmax = max([t.current for t in arr if hasattr(t, "current") and t.current is not None] + [-999])
                    if tmax != -999:
                        flat.append(f"{k}:{int(tmax)}°C")
                if flat:
                    line.append("Temps: " + ", ".join(flat))
            if line:
                log("[stats] " + " | ".join(line))
        except Exception:
            pass

# ===================== Основной цикл =====================

def main():
    add_console_handler()

    parser = argparse.ArgumentParser(description="Анализ видео с извлечением шумных фрагментов. Источник: папка/zip/rar.")
    parser.add_argument("source", nargs="?", default=VIDEO_SOURCE, help="Путь к папке или ZIP/RAR архиву с видео")
    parser.add_argument("--seven_zip", default=SEVEN_ZIP, help="Путь к 7z.exe (для .rar)")
    parser.add_argument("--segment_mode", choices=["fast_copy", "h264"], default=SEGMENT_MODE, help="Режим нарезки видео")
    parser.add_argument("--output_mode", choices=["new_root","purge"], default=OUTPUT_MODE, help="Поведение папок результата")
    parser.add_argument("--no_stats", action="store_true", help="Не показывать монитор ресурсов")
    args = parser.parse_args()

    source = args.source
    seven_zip_exe = args.seven_zip
    segment_mode = args.segment_mode
    out_mode = args.output_mode

    # Построим корневую папку результата и подключим лог-файл
    output_root = build_output_root(BASE_OUTPUT, out_mode)
    add_file_handler(os.path.join(output_root, "analyzer.log"))

    log(f"[start] Результаты: {output_root}")
    log(f"[start] Источник: {source}")
    log(f"[start] Сегментация: {segment_mode} | режим папок: {out_mode}")
    log(f"[start] Параметры: WIN={WIN_MS}ms HOP={HOP_MS}ms baseline={BASELINE_METHOD or 'calib'} "
        f"+Δ={THRESH_DELTA_DB}дБ hyst={HYST_DB}дБ keep={KEEP_SILENCE_MS}ms expand={EXTRA_EVENT_EXPAND_MS}ms merge_gap={MERGE_GAP_MS}ms")

    # Соберём список работ (для прогресса) без извлечения файлов целиком
    if os.path.isdir(source):
        worklist = collect_worklist_from_folder(source)
    elif source.lower().endswith(".zip"):
        worklist = collect_worklist_from_zip(source)
    elif source.lower().endswith(".rar"):
        if not os.path.isfile(seven_zip_exe):
            loge(f"Не найден 7-Zip: {seven_zip_exe}. Установите 7-Zip и передайте путь через --seven_zip")
            return
        worklist = collect_worklist_from_rar(source, seven_zip_exe)
    else:
        loge("SOURCE должен быть папкой или .zip/.rar архивом")
        return

    total = len(worklist)
    if total == 0:
        log("[i] Подходящих видео не найдено.")
        return
    log(f"[plan] Найдено видеофайлов: {total}")

    # CSV со сводкой по всем событиям
    csv_path = os.path.join(output_root, "events_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as csv:
        csv.write("video,event_idx,start_ms,end_ms,duration_ms,start_tc,end_tc,baseline_db,th_up_db\n")

    # Запуск потока датчиков
    stats_stop = threading.Event()
    stats_thr = None
    if SHOW_SYSTEM_STATS and not args.no_stats:
        stats_thr = threading.Thread(target=sensors_thread, args=(stats_stop,), daemon=True)
        stats_thr.start()

    # Для оценки времени
    start_all = time.perf_counter()
    per_file_durations = []

    for idx, (kind, payload, display_name) in enumerate(worklist, start=1):
        # Прогресс (до извлечения)
        if SHOW_PROGRESS:
            avg = (sum(per_file_durations) / len(per_file_durations)) if per_file_durations else None
            remaining = (total - (idx - 1)) * avg if avg else None
            log(f"[progress] Файл {idx}/{total} | осталось ~ {fmt_td(remaining)} | сейчас: {display_name}")

        # Извлечём ОДИН файл (если архив) или возьмём путь (если папка)
        tmp_dir = None
        try:
            if kind == "folder":
                video_path = payload
            elif kind == "zip":
                log(f"[extract] ZIP → {payload}")
                video_path, tmp_dir = extract_one_from_zip(source, payload)
            elif kind == "rar":
                log(f"[extract] RAR → {payload}")
                video_path, tmp_dir = extract_one_from_rar(source, payload, seven_zip_exe)
            else:
                continue

            t0 = time.perf_counter()
            filename = os.path.basename(video_path)
            video_name = os.path.splitext(filename)[0]
            out_dir = os.path.join(output_root, video_name)
            ensure_clean_video_dir(out_dir, out_mode)

            # Копия исходника (если нужно)
            if SAVE_ORIGINAL_VIDEO:
                try:
                    log("[copy] Сохраняю копию исходного видео")
                    shutil.copy2(video_path, os.path.join(out_dir, filename))
                except Exception as e:
                    logw(f"Не удалось скопировать исходник: {e}")

            log(f"[analyze] {filename}: извлекаю оригинальное аудио (WAV)")
            original_wav = os.path.join(out_dir, "original_audio.wav")
            ffmpeg_extract_wav_mono(video_path, original_wav)

            if ORIGINAL_AUDIO_SAVE in ("mp3", "both"):
                try:
                    log("[audio] Конвертирую оригинальное аудио в MP3 для прослушивания")
                    mp3_path = os.path.join(out_dir, "original_audio.mp3")
                    ffmpeg_save_mp3_from_wav(original_wav, mp3_path, bitrate=ORIGINAL_AUDIO_MP3_BITRATE)
                except Exception as e:
                    logw(f"Не удалось сохранить MP3: {e}")

            log("[analyze] Считаю кратковременный уровень (RMS)")
            audio = AudioSegment.from_wav(original_wav).set_channels(1)
            times_sec, rms_db = short_time_rms_dbfs(audio, WIN_MS, HOP_MS)

            # Бейзлайн
            if CALIBRATION_SEC > 0:
                calib_ms = min(len(audio), int(CALIBRATION_SEC * 1000))
                _, rms_calib = short_time_rms_dbfs(audio[:calib_ms], WIN_MS, HOP_MS)
                baseline_db = median(rms_calib) if rms_calib else -60.0
                baseline_src = f"first {CALIBRATION_SEC}s (median)"
            else:
                if BASELINE_METHOD.lower() == "median":
                    baseline_db = median(rms_db) if rms_db else -60.0
                    baseline_src = "global median"
                else:
                    baseline_db = percentile(rms_db, 20) if rms_db else -60.0
                    baseline_src = "global p20"

            up_th = baseline_db + THRESH_DELTA_DB
            log(f"[analyze] baseline = {baseline_db:.1f} dBFS  [{baseline_src}] ; порог = {up_th:.1f} dBFS")

            # Детекция
            log("[detect] Ищу превышения над порогом…")
            events = detect_exceedances(
                rms_db=rms_db,
                hop_ms=HOP_MS,
                baseline_db=baseline_db,
                delta_db=THRESH_DELTA_DB,
                hyst_db=HYST_DB,
                min_event_ms=MIN_EVENT_MS
            )

            # Поля и склейка
            events = pad_and_merge(events, KEEP_SILENCE_MS, MERGE_GAP_MS)
            # Доп. удлинение после склейки
            events = expand_chunks(events, EXTRA_EVENT_EXPAND_MS, total_ms=len(audio))

            if not events and rms_db:
                max_i = max(range(len(rms_db)), key=lambda i: rms_db[i])
                center_ms = max_i * HOP_MS
                half = int(FALLBACK_SEC * 1000 / 2)
                start_ms = max(0, center_ms - half)
                end_ms = min(len(audio), center_ms + half)
                events = [[start_ms, end_ms]]
                log("[detect] событий не найдено — сохранен самый громкий ~2с сегмент (fallback)")

            log(f"[export] Найдено событий: {len(events)} — начинаю экспорт аудио/видео и таймкодов")
            tc_path = os.path.join(out_dir, "timecodes.txt")
            with open(tc_path, "w", encoding="utf-8") as tc, open(csv_path, "a", encoding="utf-8") as csv:
                tc.write(f"# baseline_dBFS={baseline_db:.2f}, up_threshold={up_th:.2f} dBFS, hyst={HYST_DB:.2f} dB\n")
                tc.write(f"# win={WIN_MS}ms, hop={HOP_MS}ms, keep={KEEP_SILENCE_MS}ms, expand={EXTRA_EVENT_EXPAND_MS}ms, merge_gap={MERGE_GAP_MS}ms\n\n")

                for i, (start_ms, end_ms) in enumerate(events, 1):
                    dur_ms = end_ms - start_ms
                    start_tc = ms_to_hhmmss_ms(start_ms)
                    end_tc   = ms_to_hhmmss_ms(end_ms)
                    log(f"[export] event_{i:03}: {start_tc} → {end_tc}  (~{dur_ms/1000:.2f}s)")

                    # аудио события
                    audio[start_ms:end_ms].export(os.path.join(out_dir, f"event_{i:03}.wav"), format="wav")

                    # видео события
                    start_s = start_ms / 1000.0
                    end_s = end_ms / 1000.0
                    if segment_mode.lower() == "fast_copy":
                        out_path = os.path.join(out_dir, f"event_{i:03}.avi")
                        ffmpeg_cut_copy(video_path, out_path, start_s, end_s)
                    else:
                        out_path = os.path.join(out_dir, f"event_{i:03}.mp4")
                        ffmpeg_cut_h264(video_path, out_path, start_s, end_s, FAST_MODE)

                    # таймкод и CSV-строка
                    tc.write(f"event_{i:03}: {start_tc} - {end_tc}\n")
                    csv.write(f"{video_name},{i},{start_ms},{end_ms},{dur_ms},{start_tc},{end_tc},{baseline_db:.2f},{up_th:.2f}\n")

            # Экономия по аудио
            if ORIGINAL_AUDIO_SAVE == "mp3":
                try:
                    if os.path.exists(original_wav):
                        os.remove(original_wav)
                        log("[cleanup] WAV оригинального аудио удалён (оставлен MP3)")
                except Exception:
                    pass

            # Архивация результата
            if ARCHIVE_RESULTS:
                zip_name = f"{video_name}_results.zip"
                zip_path = os.path.join(output_root, zip_name)
                log("[zip] Упаковываю результат в ZIP…")
                zip_dir(out_dir, zip_path)
                log(f"[zip] Готово: {zip_path}")
                if DELETE_RESULT_DIR_AFTER_ZIP:
                    try:
                        shutil.rmtree(out_dir)
                        log("[zip] Папка результата удалена для экономии места")
                    except Exception as e:
                        logw(f"Не удалось удалить папку результата: {e}")

        except Exception as e:
            loge(f"Ошибка на {display_name}: {e}")
            loge(traceback.format_exc())

        finally:
            # удалим временную директорию, если файл извлекался из zip/rar
            if tmp_dir and os.path.isdir(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

            # учтём время файла для оценки ETA
            per_file_durations.append(time.perf_counter() - t0 if 't0' in locals() else 0)

            # Прогресс (после файла)
            if SHOW_PROGRESS:
                done = idx
                left = total - done
                avg = (sum(per_file_durations) / len(per_file_durations)) if per_file_durations else None
                eta = left * avg if avg else None
                spent = time.perf_counter() - start_all
                log(f"[progress] Готово {done}/{total} | прошло {fmt_td(spent)} | осталось ~ {fmt_td(eta)}")

    # Остановим поток датчиков
    if 'stats_stop' in locals():
        stats_stop.set()
    if 'stats_thr' in locals() and stats_thr:
        stats_thr.join(timeout=2)

    log("✅ Готово.")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        # Фатальная ошибка вне цикла — тоже попадёт в лог
        add_console_handler()
        print("Фатальная ошибка:", ex)
        traceback.print_exc()
