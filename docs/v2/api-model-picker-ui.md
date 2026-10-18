# Provider model choices in existing teacher settings

The model field in the existing API-teacher settings remains manually
editable. Configured providers may populate suggestions through the typed
model-discovery bridge. Discovery never supplies a key, selects a model,
starts a teacher or replaces OmniCortex. Users add their own key through
installation-private secure settings; credentials are not brain data.

Provider, credential revision and configured state define the view owner.
Changing that owner clears old suggestions and retires matching requests.
Late replies cannot repopulate a different owner. Strict Mode cleanup retires
the pending reference before cancelling, so a fresh mount can issue its own
request rather than becoming stuck behind the retired request.

The field supports native suggestions and a collapsed explicit chooser with
full-catalog search. Options are presented in pages of 100 for display only;
the complete returned catalogue remains searchable and pageable. A user must
choose an option explicitly. Refresh and Stop apply only to discovery, and
the manual field remains editable while discovery is pending. A listed model
is not claimed to support generation for the requested account or endpoint.

## Focused evidence

Three initial presentation cases passed in 21 ms: exact complete-result
ownership, invalid/cross-provider identifier rejection, and preserving the
manual value without API effects when unconfigured. A fourth full-catalog
search/paging case passed in 3 ms with the earlier three skipped. One further
late-Stop ownership case passed in 2 ms with the earlier four skipped. Stop
status, errors and finalization use the exact owner plus request, so a delayed
old cancellation reply cannot update a newer request in the same mounted view.
Renderer TypeScript passed after these final refinements.

Computer use exercised the actual component in Strict Mode with synthetic main
responses. It observed no unconfigured request, owned cleanup/cancellation and
fresh discovery, an unchanged manual value after discovery, explicit selection
from the chooser, and a manually edited value preserved through a held Refresh
and Stop. Credential-owner retirement cleared stale choices. These are
synthetic metadata operations, not real provider requests.

`test-results/api-model-picker-ui-manual/01-light-native-dropdown.jpg` is
diagnostic evidence from before the explicit chooser was added: the embedded
test browser did not expose the native suggestion popup. Subsequent explicit
chooser behavior was inspected through accessibility state. A final bounded
visual check then recorded the actual explicit chooser:

1. `02-light-explicit-chooser.jpg`: healthy narrow light entry and opened
   chooser, with manual model unchanged. The lower card continues below the
   viewport, so this image alone is not a complete below-fold check.
2. `03-dark-explicit-chooser-focused.jpg`: healthy narrow dark lower-card
   view, with the search focus ring, explicit native selector, compatibility
   warning, key-storage placeholder and manual-value receipt visible. Measured
   document/viewport width was 360 pixels and picker width was 285 pixels.

Both exact saved screenshots were reopened before acceptance. The setup used
only a synthetic configured-key flag, never an entered credential. These are
component checks, not full visual/native Electron/provider qualification. The
temporary browser tab was closed and viewport reset; it is not a replacement
application. Full theme/desktop/error/zoom and real-provider acceptance remain
separate gates.

The backend's separate fake-HTTP tests and credential boundaries are documented
in [API teacher model discovery](api-teacher-model-discovery.md). No real key,
provider, generation, training, model loading or compatibility probe ran here.
