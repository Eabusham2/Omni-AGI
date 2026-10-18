# Media teacher response contracts

Official response schemas checked October 18, 2026. This is a parsing correction
to the existing typed media lesson adapter, with no provider, model or training
execution.

OpenAI Responses and Gemini Interactions media lessons require an explicit
`status: "completed"`. Missing, pending, incomplete, failed or action-required
status cannot establish a completed lesson merely because inline bytes exist.
Provider error records are refused. OpenAI image calls must also be completed
and free of a reported error; incomplete message text cannot accompany a
successful image as though the whole lesson succeeded.

Gemini's documented `thought` steps may occur among `model_output` steps. The
adapter accepts their typed metadata while excluding their summaries from
lesson text and media. Only ordered `model_output.content` contributes parts.
Thought-only media summaries do not satisfy a request for an image or audio.
Tool calls, unsupported instructions and errored output steps remain refused.

OpenAI `output_text` may include its documented `logprobs` array. The adapter
accepts that metadata while retaining the actual output text. Gemini text does
not acquire OpenAI-specific fields. Raw response custody and downstream
base64, MIME, container, resource and asset-hash checks remain in the existing
media lesson pipeline.

Sources read:

- [OpenAI Responses schemas](https://developers.openai.com/api/reference/typescript/resources/responses)
  describe response completion states and output-message text metadata.
- [OpenAI image generation](https://developers.openai.com/api/docs/guides/image-generation#generate-images)
  describes actual `image_generation_call.result` image data.
- [OpenAI image-call schema](https://developers.openai.com/api/reference/typescript/__sdk_schema?declaration=%28resource%29+responses+%3E+%28model%29+response_output_item+%3E+%28schema%29&selected=%28resource%29+responses)
  links the image-call completion and error fields.
- [Gemini Interactions reference](https://ai.google.dev/api/interactions-api)
  specifies interaction status, errors, thought steps and model-output content.
- [Gemini streaming guide](https://ai.google.dev/gemini-api/docs/streaming)
  distinguishes thought content from final model-output content.

Six new pure wire-shape cases passed in 4 ms using only synthetic objects. They
cover explicit completion, Gemini thought/media ordering and fault refusal,
OpenAI text metadata and image/message status, and malformed metadata. The
existing media fixtures were updated to include completed status without
rerunning the closed media, text, archive or repository suites. No account,
credential, provider request, media decoder, neural model, Load, training or
hardware qualification ran. Real provider compatibility is not proved by these
object fixtures.
