import unittest

from src.app.theme import COLORS, theme_css


class AppThemeTest(unittest.TestCase):
    def test_theme_css_forces_readable_global_text_color(self):
        css = theme_css()

        self.assertIn(f"--text: {COLORS['text']};", css)
        self.assertIn(".stApp, .stApp p, .stApp span, .stApp label", css)
        self.assertIn("color: var(--text);", css)

    def test_theme_css_overrides_common_streamlit_controls(self):
        css = theme_css()

        self.assertIn("[data-testid=\"stMarkdownContainer\"]", css)
        self.assertIn("[data-baseweb=\"select\"]", css)
        self.assertIn("input, textarea", css)
        self.assertIn("color: var(--text) !important;", css)


if __name__ == "__main__":
    unittest.main()
