"""
Bottom-center floating overlay showing dictation state.

A borderless, non-activating NSPanel that floats above all windows,
is click-through (ignores mouse events), and appears only when there's
something to show. Hidden when idle.

States:
    show_recording()    red dot + "Listening…"
    show_processing()   amber glyph + "Transcribing…"
    show_done(text)     green check + truncated text; auto-hides
    hide()

All methods must be called from the main (AppKit) thread.
"""

from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSTextField,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSTimer

PANEL_WIDTH   = 240
PANEL_HEIGHT  = 44
BOTTOM_MARGIN = 100   # px above the screen bottom (clears the dock)
AUTO_HIDE_S   = 1.6
NS_STATUS_WINDOW_LEVEL = 25   # NSStatusWindowLevel — above most app windows

# (r, g, b, a) — AppKit uses 0..1 floats
C_RECORDING = (0.95, 0.30, 0.30, 1.0)   # red
C_PROCESS   = (0.95, 0.70, 0.20, 1.0)   # amber
C_DONE      = (0.30, 0.75, 0.45, 1.0)   # green
C_TEXT      = (1.00, 1.00, 1.00, 0.95)  # near-white on HUD background


def _rgb(rgba):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgba)


class DictationOverlay:
    def __init__(self):
        screen = NSScreen.mainScreen().frame()
        x = (screen.size.width - PANEL_WIDTH) / 2.0
        y = BOTTOM_MARGIN
        frame = NSMakeRect(x, y, PANEL_WIDTH, PANEL_HEIGHT)

        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False,
        )
        panel.setLevel_(NS_STATUS_WINDOW_LEVEL)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle
        )

        # Frosted glass pill
        content = NSVisualEffectView.alloc().initWithFrame_(
            NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        )
        content.setMaterial_(NSVisualEffectMaterialHUDWindow)
        content.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        content.setState_(NSVisualEffectStateActive)
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(PANEL_HEIGHT / 2.0)
        content.layer().setMasksToBounds_(True)

        dot = NSTextField.alloc().initWithFrame_(
            NSMakeRect(14, (PANEL_HEIGHT - 20) / 2.0, 20, 20)
        )
        dot.setEditable_(False)
        dot.setSelectable_(False)
        dot.setBordered_(False)
        dot.setDrawsBackground_(False)
        dot.setFont_(NSFont.boldSystemFontOfSize_(16))
        content.addSubview_(dot)

        label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(40, (PANEL_HEIGHT - 18) / 2.0, PANEL_WIDTH - 54, 18)
        )
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setBordered_(False)
        label.setDrawsBackground_(False)
        label.setFont_(NSFont.systemFontOfSize_(13))
        label.setTextColor_(_rgb(C_TEXT))
        content.addSubview_(label)

        panel.setContentView_(content)

        self._panel = panel
        self._dot = dot
        self._label = label
        self._hide_timer = None
        self._visible = False

    # ── Public API ───────────────────────────────────────────────────────────

    def show_recording(self):
        self._cancel_hide()
        self._set_state("●", C_RECORDING, "Listening…")
        self._show()

    def show_processing(self):
        self._cancel_hide()
        self._set_state("◐", C_PROCESS, "Transcribing…")
        self._show()

    def show_done(self, text: str):
        self._cancel_hide()
        text = (text or "").strip()
        if len(text) > 42:
            text = text[:40] + "…"
        self._set_state("✓", C_DONE, text or "Done")
        self._show()
        self._schedule_hide(AUTO_HIDE_S)

    def hide(self):
        self._cancel_hide()
        if self._visible:
            self._panel.orderOut_(None)
            self._visible = False

    # ── Internals ────────────────────────────────────────────────────────────

    def _set_state(self, glyph: str, dot_color, label_text: str):
        self._dot.setStringValue_(glyph)
        self._dot.setTextColor_(_rgb(dot_color))
        self._label.setStringValue_(label_text)

    def _show(self):
        if not self._visible:
            self._panel.orderFrontRegardless()
            self._visible = True

    def _schedule_hide(self, delay: float):
        self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            delay, False, lambda _t: self.hide(),
        )

    def _cancel_hide(self):
        if self._hide_timer is not None:
            try:
                self._hide_timer.invalidate()
            except Exception:
                pass
            self._hide_timer = None
