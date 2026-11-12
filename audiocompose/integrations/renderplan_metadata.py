"""Canonical AudioCompose metadata keys for RenderPlan handoffs."""

RENDERPLAN_PLAN_ID = "renderplan.plan_id"
RENDERPLAN_SOURCE_PLAN_ID = "renderplan.source_plan_id"
RENDERPLAN_PROFILE_ID = "renderplan.render_profile_id"
RENDERPLAN_SEGMENT_NUMBER = "renderplan.segment_number"
RENDERPLAN_EMPHASIS = "renderplan.emphasis"
RENDERPLAN_SEGMENT_ID = "renderplan.segment_id"
RENDERPLAN_SOURCE = "renderplan.source"
RENDERPLAN_PAUSE_ID = "renderplan.pause_id"
RENDERPLAN_PAUSE_REASON = "renderplan.pause_reason"
RENDERPLAN_PAUSE_SOURCE = "renderplan.pause_source"
RENDERPLAN_AUDIO_ID = "renderplan.audio_id"
RENDERPLAN_AUDIO_SOURCE = "renderplan.audio.src"
RENDERPLAN_AUDIO_OPTIONS = "renderplan.audio.options"

MANIFEST_ID = "render_manifest.manifest_id"
MANIFEST_WAV_SHA256 = "render_manifest.wav_sha256"
MANIFEST_WAV = "render_manifest.wav"
MANIFEST_SAMPLE_RATE = "render_manifest.sample_rate"
MANIFEST_CHANNELS = "render_manifest.channels"
MANIFEST_FRAMES = "render_manifest.frames"

AUDIOCOMPOSE_MEDIA_METADATA = "audiocompose.media.metadata"
AUDIOCOMPOSE_MEDIA_IDENTITY = "audiocompose.media.identity"
AUDIOCOMPOSE_MATERIALIZER = "audiocompose.materializer"
AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY = "audiocompose.speech_boundary.policy"
AUDIOCOMPOSE_SPEECH_BOUNDARY_INPUT_FRAMES = "audiocompose.speech_boundary.input_frames"
AUDIOCOMPOSE_SPEECH_BOUNDARY_OUTPUT_FRAMES = "audiocompose.speech_boundary.output_frames"
AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_FRAMES = "audiocompose.speech_boundary.removed_frames"
AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_SECONDS = "audiocompose.speech_boundary.removed_seconds"
AUDIOCOMPOSE_SPEECH_BOUNDARY_TRIM_END_SAMPLE = "audiocompose.speech_boundary.trim_end_sample"
AUDIOCOMPOSE_SPEECH_BOUNDARY_TRIM_CONFIRMED = "audiocompose.speech_boundary.trim_confirmed"
AUDIOCOMPOSE_SPEECH_BOUNDARY_TRIM_SILENCE_START_SAMPLE = (
    "audiocompose.speech_boundary.trim_silence_start_sample"
)
AUDIOCOMPOSE_SPEECH_BOUNDARY_TAIL_ESTIMATE_CONFIRMED = (
    "audiocompose.speech_boundary.tail_estimate_confirmed"
)
AUDIOCOMPOSE_SPEECH_BOUNDARY_TAIL_ESTIMATE_START_SAMPLE = (
    "audiocompose.speech_boundary.tail_estimate_start_sample"
)
AUDIOCOMPOSE_SPEECH_BOUNDARY_ESTIMATED_TAIL_FRAMES = (
    "audiocompose.speech_boundary.estimated_tail_frames"
)
AUDIOCOMPOSE_SPEECH_BOUNDARY_ESTIMATED_TAIL_SECONDS = (
    "audiocompose.speech_boundary.estimated_tail_seconds"
)
AUDIOCOMPOSE_PAUSE_TARGET_SECONDS = "renderplan.pause.target_seconds"
AUDIOCOMPOSE_PAUSE_ESTIMATED_PRECEDING_TAIL_SECONDS = (
    "audiocompose.pause.estimated_preceding_tail_seconds"
)
AUDIOCOMPOSE_PAUSE_APPLIED_TAIL_CREDIT_SECONDS = "audiocompose.pause.applied_tail_credit_seconds"
AUDIOCOMPOSE_PAUSE_GENERATED_SECONDS = "audiocompose.pause.generated_seconds"
AUDIOCOMPOSE_PAUSE_TAIL_CREDIT_POLICY = "audiocompose.pause.tail_credit_policy"
AUDIOCOMPOSE_PAUSE_TAIL_CREDIT_POLICY_SOURCE = "audiocompose.pause_tail_credit_policy"

RENDER_DIAGNOSTICS_FILENAME = "render-diagnostics.json"
VOICERENDER_DIAGNOSTICS_SCHEMA = "voicerender.diagnostics_schema"
VOICERENDER_DIAGNOSTICS_SHA256 = "voicerender.diagnostics_sha256"
VOICERENDER_EMPHASIS_POLICY = "voicerender.emphasis_policy"
VOICERENDER_DEFERRED_CAPABILITIES = "voicerender.deferred_capabilities"
VOICERENDER_REALIZED_CONTROLS = "voicerender.realized_controls"
