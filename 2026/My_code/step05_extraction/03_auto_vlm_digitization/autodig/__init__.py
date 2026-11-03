"""autodig - automatic chart digitization: a VLM reads the text (tick labels,
titles, series names / colours / markers), classical image processing finds the
geometry (frame, ticks, label positions, markers, lines) and turns pixels into data.

    spec.py       VLM prompts, Qwen3-VL reader, JSON parsing  (what the VLM is asked)
    detect.py     frame, tick marks, tick-label positions, text inside the plot
    check.py      pixel <-> VLM cross-check (tick-label counts, right axis) + focused re-ask
    calibrate.py  tick values + label positions -> pixel<->value maps (linear/log/broken)
    extract.py    legend, colour masks, markers, lines, same-style split -> data
    render.py     recreate the plot from extracted data
    evaluate.py   CSV in/out, series matching, scoring
    run.py        process_image(): everything for one image
    bars.py       bar charts (categories, grouped / hatched bars, error bars, broken / right axes)
    pie.py        pie charts (flat or 3D: slice colours, apex, perspective, angles)
    report.py     tables / figures for the notebook
  step 5 - more chart types, each with its own self-checks:
    spec_types.py chart-type vocabulary + the type-specific VLM questions
    colorbar.py   colour bars: find, read the scale, colour -> value
    heatmap.py    heatmaps (cell grid) and contour / colour maps (2-D surfaces)
    hist.py       histograms (filled, overlapping semi-transparent, step)
    boxplot.py    box plots (vertical / horizontal, filled, notched, grouped, fliers)
    stacked.py    stacked bars (vertical / horizontal, 100 %) and horizontal bars
    polar.py      polar curves, scatter and polar bars (angle zero / direction measured)
    axes3d.py     3-D axes -> projective camera from the axis lines and ticks
    plot3d.py     3-D bars, 3-D scatter, 3-D surfaces (depth only where the plot holds a cue)
    reject.py     ok / flagged / rejected gates (unsupported types, panels, failed self-checks)
    render_new.py recreate the step-5 chart types
    gallery_eval.py scoring of the synthetic gallery (gallery/)
"""
from .spec import QwenSpecReader, OpenAISpecReader, FileSpecReader, normalize, AXES_PROMPT, SERIES_PROMPT, BAR_PROMPT, PIE_PROMPT, TICKS_PROMPT, BOX_TICKS_PROMPT, PROMPT_VERSION
from .spec_types import HIST_PROMPT, BOX_PROMPT, STACK_PROMPT, HEATMAP_PROMPT, POLAR_PROMPT, THREED_PROMPT, canonical_type, SUPPORTED
from .extract import pipeline
from .render import recreate
from .run import process_image, read_spec, side_by_side
from . import evaluate, report
