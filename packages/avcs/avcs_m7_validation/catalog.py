from .contracts import Scenario

# These are benchmark definitions, not proof that all applications are installed.
# Each scenario must have a matching fixture and independently observable assertion
# before it can be enabled for a live run.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("native_file_save_as", "native_controls", "Windows Save As dialog", "Save the provided harmless fixture text to the requested trial-specific file path.", "file_exists_and_exact_contents", setup_notes="Use an isolated temporary directory per run."),
    Scenario("native_similar_buttons", "native_controls", "Controlled Win32 dialog", "Select the button labeled Apply, not the similarly named Cancel or Apply to All controls.", "expected_control_state", setup_notes="Use a purpose-built fixture with visually similar labels."),
    Scenario("native_context_menu", "native_controls", "Windows context menu fixture", "Open the context menu for the designated test item and choose Rename.", "item_name_changed_once", setup_notes="No real user files; disposable fixture only."),
    Scenario("native_dpi_reposition", "display_geometry", "DPI/monitor fixture", "Activate the target control after the fixture window is repositioned and resized.", "target_postcondition_after_reposition", setup_notes="Run at 100%, 150%, and 200% scale when available."),
    Scenario("custom_canvas_select", "custom_drawn_no_dom", "Custom-drawn canvas fixture", "Select the uniquely described shape in the canvas and report the selected shape id.", "selected_shape_id", setup_notes="The selected id must be read by an out-of-process evaluator."),
    Scenario("custom_canvas_drag", "custom_drawn_no_dom", "Custom-drawn canvas fixture", "Move the designated test node into the target region without moving adjacent nodes.", "node_position_in_target_region", setup_notes="Record both target and neighbor positions."),
    Scenario("browser_canvas_move", "browser_canvas", "Browser-based canvas fixture", "Move the labeled test card to the marked destination on the canvas.", "card_position", setup_notes="Use a locally hosted fixture; no external account or data."),
    Scenario("browser_dialog_delayed", "browser_app", "Local browser application fixture", "Change the fixture setting and wait until the delayed confirmation state appears.", "setting_value_and_confirmation", setup_notes="Inject deterministic 0–2 second transition delays."),
    Scenario("legacy_form_validation", "legacy_form", "Legacy-style form fixture", "Enter a benign test value and submit the form only if validation passes.", "record_value_and_validation_state", setup_notes="Use a local mock service; never submit to a live business system."),
    Scenario("accessibility_missing", "mixed_accessibility", "Hybrid semantic/visual fixture", "Identify and activate the custom-drawn control that has no useful UI Automation element.", "custom_control_state", setup_notes="The UIA-only mode is expected to abstain or fail safely."),
    Scenario("remote_desktop_focus_loss", "remote_desktop_simulated", "Simulated remote desktop fixture", "Perform the requested harmless action, but stop if the target session loses focus.", "action_state_and_focus_guard", setup_notes="Simulate focus loss; do not connect to production remote hosts."),
    Scenario("overlay_stale_target", "dynamic_ui", "Overlay and stale-target fixture", "Activate the named target only if it remains unobscured and at the grounded location.", "target_action_and_overlay_state", setup_notes="Insert an overlay between observation and action."),
)

SCENARIO_BY_ID = {s.scenario_id: s for s in SCENARIOS}
