# Action format (shared contract)

Every on-device model (base or fine-tuned) emits one JSON object per step. The Python harness and the iOS app (HarborKit) parse it the same way.

```json
{"tool": "reschedule_appointment", "args": {"member_id": "BM-10200", "appointment_id": "APT-7001", "new_date": "2026-11-13"}}
{"reply": "Done. Your eye exam with Dr. Desmond Weiss is now Friday, Nov 13 at 1:30 PM."}
```

## Conversation layout

Rendered with the model's own chat template (`tokenizer.apply_chat_template(..., enable_thinking=False)`; the Gemma 4 template otherwise injects a thinking token).

| Turn | Content |
| --- | --- |
| system | `prompts/system_core.txt` + `prompts/action_format.txt` + tool block (long form for base models, short form for fine-tuned students) |
| user | `[session] {json}` line, a blank line, then the member's message |
| assistant | one action JSON |
| user | `<tool_result name="get_policy">{json}</tool_result>` |
| assistant | next action JSON … until a `reply` |

Session JSON keys: `today`, `weekday`, `member_id`, `first_name`, `city`, `plan`, `policy_id`, `upcoming_appointments` (id, provider, specialty, datetime), `recent_claims` (id, provider, service_date, status).

The teacher (Claude) uses native tool use. Its `tool_use` blocks map to `{"tool", "args"}` and its final text to `{"reply"}`, so every trace is stored in this format and becomes SFT data directly. Text the teacher writes before a tool call is dropped (no reasoning in the targets).

## Limits

- Max 4 tool steps per request, then the agent must reply.
- Tool results are compact JSON (`separators=(",", ":")`, sorted keys) to keep prefill short on the phone.
