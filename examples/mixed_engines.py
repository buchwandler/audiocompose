"""One utterance plan using logical voices backed by different engines."""

import utterplan
from utterrender import Renderer

Planner = getattr(utterplan, "UtterPlanner", getattr(utterplan, "TTSPlanner"))
PlannerConfig = utterplan.PlannerConfig

text = '''
[Hello from the narrator.]{voice="narrator"}
[And this quotation uses another engine.]{voice="quote"}
'''
plan = Planner(PlannerConfig(language="en-us", document_format="ssmd")).plan(text)

with Renderer(
    bindings={
        "narrator": "kokoro:af_heart",
        "quote": "piper:en_US-lessac-medium",
    },
    sample_rate=24000,
) as renderer:
    renderer.to_wav(plan, "mixed.wav")
