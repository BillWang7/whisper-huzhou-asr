import csv
import html
import json
import random
from pathlib import Path

import gradio as gr

from src.scoring import align, normalize, score
from src.transcribe import transcribe

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SAMPLES = json.loads((EXAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))
SAMPLE_LABELS = [f"Clip {i + 1} ({'news' if s['file'].startswith('test_') else 'real speaker'})"
                 for i, s in enumerate(SAMPLES)]


def load_test_set(folder: str) -> dict:
    """A test set folder: audio/N.mp3 <-> row N of text.csv, plus precomputed baseline outputs."""
    path = EXAMPLES_DIR / folder
    with open(path / "text.csv", encoding="utf-8") as f:
        texts = [row["Text"] for row in csv.DictReader(f)]
    with open(path / "baseline.csv", encoding="utf-8") as f:
        baseline = {int(row["clip"]): row["baseline_prediction"] for row in csv.DictReader(f)}
    return {"dir": path, "folder": folder, "texts": texts, "baseline": baseline}


# Two held-out test sets, neither used in training
TEST_SETS = {
    "News broadcasts": load_test_set("test_set"),  # same channel as the training data
    "Real speakers": load_test_set("real_speakers"),  # people reading everyday sentences, mostly on phones
}
SET_NAMES = list(TEST_SETS)

MODEL_URL = "https://huggingface.co/BillWang7/whisper-medium-ch-hz"
GITHUB_URL = "https://github.com/BillWang7/whisper-huzhou-asr"


# ---------- Result rendering ----------

def placeholder(msg: str) -> str:
    return f'<div class="placeholder">{html.escape(msg)}</div>'


def score_tile(title: str, reference: str, prediction: str, kind: str) -> str:
    cer_value, correct, length = score(reference, prediction)
    return (f'<div class="tile {kind}"><div class="tile-title">{html.escape(title)}</div>'
            f'<div class="tile-num">{cer_value:.0f}%</div>'
            f'<div class="tile-lbl">character error rate</div>'
            f'<div class="tile-sub">{correct} of {length} characters correct</div></div>')


def na_tile(title: str, msg: str) -> str:
    return (f'<div class="tile na"><div class="tile-title">{html.escape(title)}</div>'
            f'<div class="tile-num">–</div><div class="tile-sub">{html.escape(msg)}</div></div>')


def diff_grid(title: str, reference: str, prediction: str, model_label: str, note: str = "") -> str:
    cols = [f'<div class="col lab"><span class="c">What was said</span><span class="c">{html.escape(model_label)}</span></div>']
    # A runaway output (the model stuck repeating a character) would draw hundreds of columns.
    # Show only its start and say how much was cut; the score above still counts all of it.
    shown, limit = normalize(prediction), 2 * len(normalize(reference)) + 10
    hidden = max(len(shown) - limit, 0)
    for r, h in align(reference, shown[:limit]):
        if r == h:
            top, bot = f'<span class="c">{html.escape(r)}</span>', f'<span class="c ok">{html.escape(h)}</span>'
        elif r is None:  # extra character the model added
            top, bot = '<span class="c gap">·</span>', f'<span class="c bad">{html.escape(h)}</span>'
        elif h is None:  # character the model missed
            top, bot = f'<span class="c miss">{html.escape(r)}</span>', '<span class="c bad">✕</span>'
        else:  # wrong character
            top, bot = f'<span class="c miss">{html.escape(r)}</span>', f'<span class="c bad">{html.escape(h)}</span>'
        cols.append(f'<div class="col">{top}{bot}</div>')
    if hidden:
        cols.append(f'<div class="col more">… {hidden} more extra characters not shown</div>')
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="section"><div class="section-title">{html.escape(title)}</div>'
            f'<div class="diff">{"".join(cols)}</div>{note_html}</div>')


LEGEND = ('<div class="legend"><span class="c ok">字</span> correct '
          '<span class="c bad">字</span> wrong or extra '
          '<span class="c bad">✕</span> missed</div>')


def comparison(reference: str, ours: str, baseline: str | None, *, english: str = "", ours_note: str = "",
               baseline_note: str = "", baseline_missing: str = "", ref_caption: str = "") -> str:
    parts = []
    if english:
        parts.append(f'<div class="meaning"><span>English meaning</span>{html.escape(english)}</div>')
    tiles = [score_tile("Fine-tuned model (this project)", reference, ours, "ours")]
    if baseline is not None:
        tiles.append(score_tile("Baseline Whisper-medium*", reference, baseline, "base"))
    else:
        tiles.append(na_tile("Baseline Whisper-medium*", baseline_missing))
    parts.append(f'<div class="tiles">{"".join(tiles)}</div>')
    parts.append(diff_grid("Fine-tuned model vs. what was said", reference, ours, "Our model", ours_note))
    if baseline is not None:
        parts.append(diff_grid("Baseline Whisper vs. what was said", reference, baseline, "Baseline", baseline_note))
    parts.append(LEGEND)
    if ref_caption:
        parts.append(f'<div class="note">{html.escape(ref_caption)}</div>')
    return "".join(parts)


# ---------- Callbacks ----------

def select_sample(label: str):
    s = SAMPLES[SAMPLE_LABELS.index(label)]
    return str(EXAMPLES_DIR / s["file"]), placeholder("Press Transcribe to compare the models.")


def run_sample(label: str) -> str:
    s = SAMPLES[SAMPLE_LABELS.index(label)]
    ours = transcribe(str(EXAMPLES_DIR / s["file"]))
    if ours == s["expected_finetuned"] and s["finetuned_note"]:
        ours_note = s["finetuned_note"]
    elif score(s["reference"], ours)[0] == 0:
        ours_note = "Exact match with what was said."
    else:
        ours_note = ""
    return comparison(s["reference"], ours, s["baseline"], english=s["english"], ours_note=ours_note,
                      baseline_note=s.get("baseline_note", "Gibberish: it has nothing to do with what was said."))


def test_clip_path(set_name: str, n) -> str:
    return str(TEST_SETS[set_name]["dir"] / "audio" / f"{int(n)}.mp3")


def select_set(set_name: str):
    return gr.update(maximum=len(TEST_SETS[set_name]["texts"]) - 1, value=0)


def select_test(set_name: str, n):
    return test_clip_path(set_name, n), placeholder(f"{set_name} clip #{int(n)} loaded. Press Transcribe.")


def random_test(set_name: str) -> int:
    return random.randrange(len(TEST_SETS[set_name]["texts"]))


def run_test(set_name: str, n) -> str:
    n, t = int(n), TEST_SETS[set_name]
    return comparison(t["texts"][n], transcribe(test_clip_path(set_name, n)), t["baseline"].get(n),
                      baseline_missing="Not computed for this clip",
                      ref_caption=f"{set_name} clip #{n}. Reference transcript from examples/{t['folder']}/text.csv.")


def run_own(audio_path: str | None) -> str:
    if not audio_path:
        return placeholder("Add some audio first.")
    text = transcribe(audio_path)
    return (f'<div class="section"><div class="section-title">Fine-tuned model output</div>'
            f'<div class="own-text">{html.escape(text) or "<em>(no speech detected)</em>"}</div>'
            '<div class="note">There\'s no reference transcript for your own audio, so no error rate is shown.</div></div>')


# ---------- Layout ----------

CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; }
#header h1 { margin-bottom: 0.2em; }
.stats { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 4px; }
.stat { flex: 1 1 180px; padding: 12px 16px; border-radius: 12px;
        border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.stat .set { font-weight: 600; font-size: 0.9em; opacity: 0.85; }
.stat .pair { display: flex; align-items: baseline; gap: 8px; margin: 2px 0; }
.stat .num { font-size: 1.6em; font-weight: 700; }
.stat .num.good { color: #16a34a; }
.stat .num.bad { color: #dc2626; }
.stat .vs { opacity: 0.6; }
.stat .lbl { font-size: 0.85em; opacity: 0.75; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px; }
.tile { padding: 16px 20px; border-radius: 14px; border: 1px solid var(--border-color-primary);
        background: var(--block-background-fill); border-top-width: 6px; }
.tile.ours { border-top-color: #16a34a; }
.tile.base { border-top-color: #dc2626; }
.tile.na   { border-top-color: #9ca3af; border-style: dashed; }
.tile-title { font-weight: 600; font-size: 0.95em; opacity: 0.85; }
.tile-num { font-size: 3.6em; font-weight: 800; line-height: 1.05; margin-top: 6px; }
.tile.ours .tile-num { color: #16a34a; }
.tile.base .tile-num { color: #dc2626; }
.tile.na .tile-num { opacity: 0.4; }
.tile-lbl { font-size: 0.9em; font-weight: 600; opacity: 0.75; }
.tile-sub { margin-top: 6px; font-size: 0.9em; opacity: 0.75; }
.section { padding: 14px 16px; margin-bottom: 12px; border-radius: 12px;
           border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.section-title { font-weight: 600; font-size: 0.9em; opacity: 0.85; margin-bottom: 10px; }
.diff { display: flex; flex-wrap: wrap; row-gap: 10px; }
.col { display: inline-flex; flex-direction: column; align-items: center; }
.col.lab { align-items: flex-start; margin-right: 10px; }
.col.more { justify-content: flex-end; font-size: 0.85em; opacity: 0.7; padding: 0 0 4px 8px; }
.col.lab .c { font-size: 0.8em; font-weight: 600; opacity: 0.6; min-width: 0; line-height: 2.2; }
.c { font-size: 1.4em; line-height: 1.55; min-width: 1.3em; text-align: center; padding: 0 2px; border-radius: 5px; margin: 1px; }
.c.ok { color: #16a34a; }
.c.bad { background: rgba(220,38,38,0.16); color: #dc2626; font-weight: 700; }
.c.miss { text-decoration: underline 2px #dc2626; text-underline-offset: 4px; }
.c.gap { opacity: 0.3; }
.legend { font-size: 0.85em; opacity: 0.8; margin: 4px 0 8px; }
.legend .c { font-size: 1.05em; margin: 0 2px 0 10px; }
.legend .c:first-child { margin-left: 0; }
.own-text { font-size: 1.4em; line-height: 1.6; }
.note { margin-top: 8px; font-size: 0.85em; opacity: 0.7; }
.meaning { margin-bottom: 12px; padding: 10px 16px; border-radius: 12px; font-size: 1.05em;
           background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.meaning span { display: block; font-size: 0.75em; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.04em; opacity: 0.6; margin-bottom: 2px; }
.placeholder { padding: 18px; text-align: center; opacity: 0.6; border: 1px dashed var(--border-color-primary); border-radius: 12px; }
"""

HEADER = f"""
<div id="header">
<h1>Huzhou Dialect Speech Recognition</h1>
<p>Whisper-medium fine-tuned on ~15,000 clips of Huzhou dialect (湖州话), a Wu Chinese dialect that
standard speech recognition handles poorly. Output is written in Mandarin characters.
<a href="{MODEL_URL}" target="_blank">Model on Hugging Face</a> · <a href="{GITHUB_URL}" target="_blank">Code on GitHub</a></p>
<div class="stats">
  <div class="stat"><div class="set">News broadcasts · 1,705 held-out clips</div>
    <div class="pair"><span class="num good">26.7%</span><span class="vs">vs</span><span class="num bad">98.8%</span></div>
    <div class="lbl">character error rate: fine-tuned vs baseline Whisper</div></div>
  <div class="stat"><div class="set">Real speakers · 828 everyday sentences</div>
    <div class="pair"><span class="num good">55.6%</span><span class="vs">vs</span><span class="num bad">93.5%</span></div>
    <div class="lbl">character error rate: fine-tuned vs baseline Whisper</div></div>
</div>
</div>
"""

with gr.Blocks(title="Huzhou Dialect ASR", css=CSS, theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.HTML(HEADER)

    with gr.Tabs():
        with gr.Tab("🎧 Featured clips"):
            gr.Markdown("Hand-picked Huzhou dialect clips the model never trained on, with English translations: "
                        "news broadcasts, and real speakers reading everyday sentences.")
            sample_pick = gr.Radio(SAMPLE_LABELS, value=SAMPLE_LABELS[0], label="Sample clips")
            sample_audio = gr.Audio(value=str(EXAMPLES_DIR / SAMPLES[0]["file"]), type="filepath",
                                    interactive=False, label="Audio")
            sample_btn = gr.Button("Transcribe", variant="primary")
            sample_out = gr.HTML(placeholder("Press Transcribe to compare the models."))
            sample_pick.change(select_sample, sample_pick, [sample_audio, sample_out])
            sample_btn.click(run_sample, sample_pick, sample_out)

        with gr.Tab("🎲 Random test clip"):
            gr.Markdown("Don't trust hand-picked examples? Roll a random clip (good and bad) from either held-out "
                        "test set, or choose one by number. **News broadcasts** are from the same TV channel as the "
                        "training data; **real speakers** are people reading everyday sentences, mostly recorded on "
                        "phones, which is harder. Clips and transcripts are in `examples/test_set/` and "
                        "`examples/real_speakers/`.")
            test_set = gr.Radio(SET_NAMES, value=SET_NAMES[0], label="Test set")
            with gr.Row(equal_height=True):
                test_num = gr.Slider(0, len(TEST_SETS[SET_NAMES[0]]["texts"]) - 1, value=0, step=1,
                                     label="Clip number", scale=4)
                rand_btn = gr.Button("🎲 Random clip", scale=1)
            test_audio = gr.Audio(value=test_clip_path(SET_NAMES[0], 0), type="filepath", interactive=False,
                                  label="Test clip")
            test_btn = gr.Button("Transcribe", variant="primary")
            test_out = gr.HTML(placeholder("Pick a clip, then press Transcribe."))
            test_set.change(select_set, test_set, test_num).then(select_test, [test_set, test_num], [test_audio, test_out])
            test_num.change(select_test, [test_set, test_num], [test_audio, test_out])
            rand_btn.click(random_test, test_set, test_num)
            test_btn.click(run_test, [test_set, test_num], test_out)

        with gr.Tab("🎙️ Your own audio"):
            gr.Markdown("Upload a recording of Huzhou dialect (MP3, WAV, M4A…), or switch to the microphone "
                        "to record yourself. Clips under 30 seconds work best.")
            own_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Your audio")
            own_btn = gr.Button("Transcribe", variant="primary")
            own_out = gr.HTML(placeholder("Upload or record something, then press Transcribe."))
            own_btn.click(run_own, own_audio, own_out)

    gr.Markdown(
        "<small>\\* <b>Baseline outputs are precomputed, not run live.</b> They were generated once in advance by "
        "running the unmodified <code>openai/whisper-medium</code> on every test clip, with the same settings "
        "(language: Chinese, task: transcribe, same loop guard), and saved with the demo, so you don't need to download a second model. "
        "The fine-tuned model's output is computed live every time you press Transcribe.<br>"
        "Runs on CPU, so a transcription can take several seconds. "
        "Error rates are character error rate (CER): edit distance divided by reference length, ignoring punctuation, "
        "which can exceed 100% when the output is longer than the reference. "
        "All featured and random clips are held out: none were used in training. Real-speaker sentences come "
        "from everyday dialogue and narration, so word choices the model renders differently count as errors "
        "even when the meaning is kept.</small>"
    )

if __name__ == "__main__":
    demo.launch()
